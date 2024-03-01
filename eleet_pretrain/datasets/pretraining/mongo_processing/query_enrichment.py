"""Combine data loaded by the loaders."""

import logging
import multiprocessing

from eleet_pretrain.datasets.pretraining.mongo_processing.text_preparation import TextPreparationStep  # noqa
from eleet_pretrain.datasets.pretraining.mongo_processing.query_preparation import QueryPreparationStep  # noqa
from eleet_pretrain.datasets.pretraining.data_import.wikidata_utils import shorten_uri
from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.steps import Step
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import SUBJ_INDEX
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI, SUBJ_INDEX, DOC_INDEX
from eleet_pretrain.datasets.pretraining.mongo_processing.mongo_store import MongoStore, multiprocessing_get_ranges, \
    multiprocessing_mongo_clean, multiprocessing_mongo_rename, multiprocessing_update_process_bar

from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI, DOC_URI, SUBJ_INDEX, DOC_INDEX, \
    ATTR_DESCRIPTION, ATTR_LABEL, DOC_NUMBER, SUBJECT_URI, HINT_OBJ, HINT_PRED_LABEL, HINT_PRED_URI, ANSWER_START, \
        ANSWER_END, ANSWER_URI, ANSWER_SURFACEFORM, ANSWER_NORMALIZED

logger = logging.getLogger(__name__)

TMP_SPLIT_COLLECTIONS = ["tmp-split-1", "tmp-split-2"]
ATTR_COLS = [ATTR_LABEL, ATTR_URI, ATTR_DESCRIPTION]
DOC_COLS = [DOC_URI, DOC_NUMBER, DOC_INDEX]
SUBJECT_COLS = [SUBJECT_URI, SUBJ_INDEX]
HINT_COLS = [HINT_OBJ, HINT_PRED_URI, HINT_PRED_LABEL]
ANSWER_COLS = [ANSWER_START, ANSWER_END, ANSWER_URI, ANSWER_SURFACEFORM, ANSWER_NORMALIZED]

# TODO rename back directly after each operation
TMP_COLLECTIONS = [f"tmp-query-enrichment-collection-{i}" for i in range(6)]
TMP_TOKEN_OVERLAP_COLLECTION = "tmp-token-overlap-collection"


class QueryEnrichment(BaseLoader):
    """Merge data from the different sources."""

    def __init__(self, *args, do_add_evidence=True, do_add_texts=True, do_group_into_tables=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_add_texts = do_add_texts
        self.do_add_evidence = do_add_evidence
        self.do_group_into_tables = do_group_into_tables

    def enrich_queries(self, num_workers):
        logger.info("Enrich queries")
        multiprocessing_mongo_clean(*TMP_COLLECTIONS, TMP_TOKEN_OVERLAP_COLLECTION)
        if self.do_add_texts:
            logger.info("Add texts.")
            self.multiprocessing_preprocess(
                data_loader_process=multiprocessing_get_ranges(self.prepared_query_stores, 30_000),
                writer_process=multiprocessing_update_process_bar(self.prepared_query_stores, "Add texts"),
                worker_process=self._add_texts,
                num_workers=num_workers,
            )
            multiprocessing_mongo_rename(self.prepared_query_stores, TMP_COLLECTIONS)
        if self.do_add_evidence:
            logger.info("Add evidence.")
            relevant_predicates = self.compute_relevant_predicates()
            self.multiprocessing_preprocess(
                data_loader_process=multiprocessing_get_ranges(self.prepared_query_stores, 30_000),
                writer_process=multiprocessing_update_process_bar(self.prepared_query_stores, "Add evidence"),
                worker_process=self._add_evidence,
                num_workers=num_workers,
                worker_args=(relevant_predicates, ),
            )
            multiprocessing_mongo_rename(self.prepared_query_stores, TMP_COLLECTIONS)
        if self.do_group_into_tables:
            logger.info("Group into tables.")
            relevant_predicates = self.compute_relevant_predicates()
            self.multiprocessing_preprocess(
                data_loader_process=multiprocessing_get_ranges(self.prepared_query_stores, 30_000),
                writer_process=multiprocessing_update_process_bar(self.prepared_query_stores, "Group into tables"),
                worker_process=self._group_into_tables,
                num_workers=num_workers
            )
            multiprocessing_mongo_rename(self.prepared_query_stores, TMP_COLLECTIONS)
        multiprocessing_mongo_clean(*TMP_COLLECTIONS, TMP_TOKEN_OVERLAP_COLLECTION)

    def compute_relevant_predicates(self):
        q = multiprocessing.Queue()
        p1 = multiprocessing.Process(target=self._compute_relevant_predicates, daemon=True, args=(q,))
        p1.start()
        p1.join()
        relevant_predicates = q.get()
        return relevant_predicates

    def _compute_relevant_predicates(self, q):
        self.mongo_connect()
        q.put(set(shorten_uri(x) for x in self.query_store.distinct("_attr_uri")))

    def _add_texts(self, job_queue, example_queue, worker_id):
        self.mongo_connect()
        job = job_queue.get()
        while job is not None:
            skip, limit, total, split = job

            self.prepared_query_stores[split].aggregate([
                {"$skip": skip}, {"$limit": limit},
                {
                    "$lookup": {
                        "from": self.prepared_text_store.collection_name,
                        "localField": f"_id.{DOC_INDEX}",
                        "foreignField": "text_idx",
                        "as": "_tmp_text0",
                        "pipeline": [
                            {"$project": {"covers": 1, "text_idx": 1, "_id": 0}}
                        ]
                    }
                }, {
                    "$project": {
                        "_queries": 1,
                        "text": {"$first": "$_tmp_text0"}
                    }
                },
                {"$merge": TMP_COLLECTIONS[split]}
            ])
            example_queue.put((limit, total, split))
            job = job_queue.get()

    def _add_evidence(self, relevant_predicates, job_queue, example_queue, worker_id):
        self.mongo_connect()
        job = job_queue.get()
        while job is not None:
            skip, limit, total, split = job

            self.prepared_query_stores[split].aggregate([
                {"$skip": skip}, {"$limit": limit},
                {
                    "$lookup": {  # evidence
                        "from": self.wikidata_store.collection_name,
                        "localField": f"_id.{DOC_INDEX}",
                        "foreignField": "wikidata_id",
                        "pipeline": [
                            {"$project": {"neighbors": 1, "wikidata_id": 1,
                                          "types": 1, "superclasses": 1}},
                            {"$limit": 1}
                        ],
                        "as": "_evidence"
                    }
                },
                {"$unwind": {"path": "$_evidence", "preserveNullAndEmptyArrays": False}},

                # To increase overlap between facts mentioned in the text and in the corresponding row,
                # we add further evidence about entities mentioned in the text, which are directly linked to the main
                # entity. E.g. The row also should also contain info about the main entity's parents, if these are
                # mentioned in the text.
                {"$set": {"_neighboring_evidence": {"$slice": [{
                    "$setIntersection": ["$text.covers.s", "$_evidence.neighbors.o"]}, 2]}}},
                {"$set": {"_neighboring_evidence": {"$map": {
                    "input": "$_neighboring_evidence",
                    "in": {"p": {"$arrayElemAt": ["$_evidence.neighbors.p",
                                                  {"$indexOfArray": ["$_evidence.neighbors.o", "$$this"]}]},
                            "o": "$$this"}
                }}}},
                {"$set": {"_evidence_types": {"$concatArrays": ["$_evidence.types", "$_evidence.superclasses"]}}},
                {"$set": {"_evidence": "$_evidence.wikidata_id"}},
                *([] if not self.sample else [{"$limit": 100}]),
                {"$merge": TMP_COLLECTIONS[split]}
            ])
            example_queue.put((limit, total, split))
            job = job_queue.get()

    def _group_into_tables(self, job_queue, example_queue, worker_id):
        """Group the different documents in small groups based on same type, subclass or common neighbors."""
        MIN_ENTITIES_WITH_TYPE = 5
        MAX_ENTITIES_WITH_TYPE = 50
        ALLOWED_TYPES = {3: ["Q17334923", "Q5"]}

        self.mongo_connect()
        job = job_queue.get()
        while job is not None:
            skip, limit, total, split = job

            # First compute recursive types and superclasses, as well as all neighbors.
            # Afterwards, we group entities with common type or superclasses together.
            # For frequent types, we also take common neighbors into account. 
            self.prepared_query_stores[split].aggregate([
                {"$skip": skip}, {"$limit": limit},
                {"$set": {"obj": "$$ROOT"}},
                {"$lookup": {  # Get subclasses, neighbors, types from wikidata
                    "from": "wikidata",
                    "localField": f"_id.{DOC_INDEX}",
                    "foreignField": "wikidata_id",
                    "as": "x",
                    "pipeline": [{"$project": {"types": 1, "superclasses": 1, "neighbors": 1}}]
                }},
                {"$project": {"types": {"$first": "$x.types"}, "superclasses": {"$first": "$x.superclasses"}, "obj": 1,
                              "neighbors": {"$first": "$x.neighbors"}, "wikidata_id": f"$_id.{DOC_INDEX}", "_id": 0}},
                {"$lookup": {
                    "from": "wikidata-class-hierarchy",  # look up recursive superclasses of the types
                    "localField": "types",
                    "foreignField": "wikidata_id",
                    "as": "x",
                    "pipeline": [{"$group": {"_id": 1,
                                             "transitive_superclasses": {"$push": "$transitive_superclasses"}}},
                                 {"$set": {"transitive_superclasses": {"$reduce": {
                                     "input": "$transitive_superclasses",
                                     "initialValue": [],
                                     "in": {"$setUnion": ["$$this", "$$value"]}
                                 }}}}]
                }},
                {"$project": {"types": {"$first": "$x.transitive_superclasses"}, "superclasses": 1,
                              "wikidata_id": 1, "neighbors": 1, "_id": 0, "obj": 1}},
                {"$lookup": {
                    "from": "wikidata-class-hierarchy",  # look up recursive superclasses
                    "localField": "wikidata_id",
                    "foreignField": "wikidata_id",
                    "as": "x"
                }},
                {"$project": {"types": 1, "superclasses": {"$first": "$x.transitive_superclasses"},
                              "wikidata_id": 1, "neighbors": 1, "_id": 0, "obj": 1}},
                *([] if split not in ALLOWED_TYPES else [
                    {"$set": {"types": {"$setIntersection": ["$types", ALLOWED_TYPES[split]]}}}
                ]),
                {"$out": f"tmp-{worker_id}"}
            ], allowDiskUse=True)

            s = MongoStore(f"tmp-{worker_id}")
            s.connect()
            num_rounds = 3
            for i, grouping_criterion in enumerate(("types", "superclasses") * num_rounds):
                final_round = (i == num_rounds * 2 - 1)


                for gc in (grouping_criterion, "neighbors"):
                    s.aggregate([  # collect types / subclasses / ... useful for matching
                        {"$unwind": f"${gc}"},
                        {"$group": {"_id": f"${gc}", "count": {"$count": {}}}},  # filter out super rare types, ...
                        {"$match": {"$expr": {"$gt": ["$count", MIN_ENTITIES_WITH_TYPE]}}},
                        {"$out": f"tmp-{gc}-{worker_id}"}
                    ], allowDiskUse=True)

                s.aggregate([
                    {"$lookup": {  # match docs
                        "from": f"tmp-{grouping_criterion}-{worker_id}",
                        "localField": f"{grouping_criterion}",
                        "foreignField": "_id",
                        "as": "x",
                        "pipeline": [{"$sort": {"count": 1}}, {"$limit": 1}]
                    }},
                    {"$lookup": {   # frequent types are grouped by common neighbor as well
                        "from": f"tmp-neighbors-{worker_id}",
                        "localField": f"neighbors",
                        "foreignField": "_id",
                        "as": "y",
                        "pipeline": [{"$sort": {"count": 1}}, {"$limit": 1}]
                    }},
                    {"$unwind": {"path": "$x", "preserveNullAndEmptyArrays": True}},
                    {"$unwind": {"path": "$y", "preserveNullAndEmptyArrays": True}},
                    {"$set": {"y": {"$switch": {"branches": [  # rare types grouped by type only
                        {"case": {"$gt": ["$x.count", MAX_ENTITIES_WITH_TYPE]}, "then": "$y"}
                    ], "default": None}}}},
                    {"$group": {"_id": {"type": "$x._id", "neighbor": "$y._id"}, "docs": {"$push": "$$ROOT"}}},
                    {"$out": f"tmp2-{worker_id}"}
                ], allowDiskUse=True)

                s2 = MongoStore(f"tmp2-{worker_id}")
                s2.connect() 
                s2.aggregate([  # matching successful, merge with final result
                    *([{"$match": {"_id": {"$ne": {}}, "docs.1": {"$exists": True}}}]
                      if not final_round else []),
                    {"$set": {"docs": "$docs.obj"}},
                    {"$set": {"_id.worker_id": worker_id, "_id.skip": skip}},
                    {"$merge": TMP_COLLECTIONS[split]}
                ])

                if not final_round:
                    s2.aggregate([  # matching failed, try to match these again
                        {"$match": {"$or": [{"_id": {"$eq": {}}}, {"docs.1": {"$exists": False}}]}},
                        {"$unwind": "$docs"}, {"$unset": "docs.x"}, {"$unset": "docs.y"},
                        {"$replaceRoot": {"newRoot": "$docs"}}, {"$out": f"tmp-{worker_id}"}
                    ])

                s2.drop()
                for gc in (grouping_criterion, "neighbors"):
                    s3 = MongoStore(f"tmp-{gc}-{worker_id}")
                    s3.connect()
                    s3.drop()
            s.drop()

            example_queue.put((limit, total, split))
            job = job_queue.get()


class QueryEnrichmentStep(Step):
    """Load data from wikidata dump."""

    depends_on = {QueryPreparationStep, TextPreparationStep}

    def check_done(self, args, dataset):
        """Check whether the step has already been executed."""
        x = QueryEnrichment(dataset, args.dataset_dir, args.small_sample)
        x.mongo_connect()
        if not x.do_group_into_tables:
            return (
                (not x.do_add_texts or "text" in x.prepared_query_stores[0].find_one()) and
                (not x.do_add_evidence or "_evidence" in x.prepared_query_stores[0].find_one())
            )
        else:
            return "docs" in x.prepared_query_stores[0].find_one() and (
                (not x.do_add_texts or "text" in x.prepared_query_stores[0].find_one()["docs"][0]) and
                (not x.do_add_evidence or "_evidence" in x.prepared_query_stores[0].find_one()["docs"][0])
            )


    def run(self, args, dataset):
        """Execute the step."""
        x = QueryEnrichment(dataset, args.dataset_dir, args.small_sample)
        x.enrich_queries(args.num_workers)
