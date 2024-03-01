"""Combine data loaded by the loaders."""

import logging
import multiprocessing
import tqdm

from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.datasets.pretraining.mongo_processing.query_preparation import QueryPreparationStep
from eleet_pretrain.steps import Step
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ANSWER_END, ANSWER_START, ANSWER_URI, SUBJ_INDEX
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI, SUBJ_INDEX, DOC_INDEX
from eleet_pretrain.datasets.pretraining.mongo_processing.mongo_store import MongoStore
from eleet_pretrain.datasets.pretraining.python_processing.utils import SPLIT_TEST_ALL

logger = logging.getLogger(__name__)


class TextPreparation(BaseLoader):
    """Merge data from the different sources."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_texts(self, num_workers):
        logger.info("Prepare texts")
        self.multiprocessing_preprocess(
            data_loader_process=self.get_ranges,
            writer_process=self.update_process_bar,
            worker_process=self._prepare_texts,
            num_workers=num_workers,
        )
        p = multiprocessing.Process(target=self.wrap_up, daemon=True, args=(num_workers,))
        p.start()
        p.join()

    def wrap_up(self, num_workers):
        for i in range(num_workers):
            s = MongoStore(f"tmp-text-{i}")
            s.connect()
            s.drop()
        self.mongo_connect()
        self.prepared_text_store.create_indexes()

    def get_ranges(self, job_queue, num_workers):
        self.mongo_connect()
        count = self.text_store.estimated_document_count()
        slice_size = 30_000
        for i in range(0, count + slice_size, slice_size):
            job_queue.put((i, slice_size, count))
        for _ in range(num_workers):
            job_queue.put(None)

    def update_process_bar(self, example_queue):
        job = example_queue.get()
        with tqdm.tqdm(total=job[-1], desc="Preparing text") as pbar:
            while job is not None:
                pbar.update(job[0])
                job = example_queue.get()


    def _prepare_texts(self, job_queue, example_queue, worker_id):
        self.mongo_connect()
        job = job_queue.get()
        while job is not None:
            skip, limit, total = job
            self.text_store.aggregate([
                {"$skip": skip}, {"$limit": limit},  # get types of texts
                {
                    "$project": { DOC_INDEX: 1 },
                },
                { "$lookup": {
                    "from": self.wikidata_store.collection_name,
                    "localField": DOC_INDEX,
                    "foreignField": "wikidata_id",
                    "pipeline": [
                        {"$project": {"types": {"$concatArrays": [
                            {"$ifNull": ['$types', []]},
                            {"$ifNull": ['$subclasses', []]}
                        ]}}},
                    ],
                    "as": "tmp_type"
                }},
                {
                    "$project": {DOC_INDEX: 1, "types": {"$first": "$tmp_type.types"}}
                },
                {"$lookup": {  # compute which wikidata attributes the text covers
                    "from": self.query_store.collection_name,
                    "localField": DOC_INDEX,
                    "foreignField": DOC_INDEX,
                    "pipeline": [
                        {"$project": {
                            "s": f"${SUBJ_INDEX}",
                            "p": {"$last": { "$split": [ f"${ATTR_URI}", "/" ] }},
                            "o": f"${ANSWER_URI}",
                            "start": f"${ANSWER_START}",
                            "end": f"${ANSWER_END}",
                            "_id": 0
                        }}
                    ],
                    "as": "covers"
                }},
                {"$lookup": {  # avoid tests from test set
                    "from": SPLIT_TEST_ALL,
                    "localField": DOC_INDEX,
                    "foreignField": "_id",
                    "as": "is_test"
                }},
                {"$set": {"is_test": {"$gt": [{"$size": "$is_test"}, 0]}}},
                {"$out": f"tmp-text-{worker_id}"}
            ])

            store = MongoStore(f"tmp-text-{worker_id}")
            store.connect()
            logger.info(f"Create Indexes for tmp-text-{worker_id}.")
            store.create_index("types")
            store.create_index("text_idx")

            store.aggregate([
                {"$lookup": {
                    "from": f"tmp-text-{worker_id}",  # TODO avoid text from test set
                    "localField": "types",
                    "foreignField": "types",
                    "let": {"this_is_test": "$is_test", "this_id": f"${DOC_INDEX}"},
                    "pipeline": [
                        {"$match": {"$expr": {"$eq": ["$is_test", "$$this_is_test"]}}},
                        {"$match": {"$expr": {"$ne": [f"${DOC_INDEX}", "$$this_id"]}}},
                        {"$sample": {"size": 1}}
                    ],
                    "as": "c"}
                },
                {"$lookup": {
                    "from": self.text_store.collection_name,
                    "localField": DOC_INDEX,
                    "foreignField": DOC_INDEX,
                    "as": "text"}
                },
                {"$lookup": {  # confusion text from entity of same type
                    "from": self.text_store.collection_name,
                    "localField": "c." + DOC_INDEX,
                    "foreignField": DOC_INDEX,
                    "as": "confusion"}
                },
                {"$set": {
                    "text": {"$first": "$text"},
                    "confusion": {"$first": "$confusion"},
                    "c": {"$first": "$c"}
                }},
                {"$project": {
                    "text": "$text._text",
                    "text_idx": "$text._doc",
                    "covers": "$covers",
                    "confusion": "$confusion._text",
                    "confusion_idx": "$confusion._doc",
                    "confusion_covers": "$c.covers",
                    "is_test": "$is_test"
                }},
                {"$merge": self.prepared_text_store.collection_name}
            ])
            example_queue.put((limit, total))
            job = job_queue.get()


class TextPreparationStep(Step):
    """Load data from wikidata dump."""

    depends_on = {QueryPreparationStep}

    def check_done(self, args, dataset):
        """Check whether the step has already been executed."""
        x = TextPreparation(dataset, args.dataset_dir, args.small_sample)
        x.mongo_connect()
        return not x.prepared_text_store.is_empty()

    def run(self, args, dataset):
        """Execute the step."""
        x = TextPreparation(dataset, args.dataset_dir, args.small_sample)
        x.prepare_texts(args.num_workers)
