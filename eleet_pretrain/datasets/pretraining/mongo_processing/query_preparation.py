"""Combine data loaded by the loaders."""

import logging
import multiprocessing

from eleet_pretrain.datasets import TRExPreprocessStep
from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.datasets.pretraining.python_processing.utils import TEST_SPLIT_DEFINITIONS, SPLIT_TEST_ALL, UNSEEN_QUERY_ATTRS
from eleet_pretrain.steps import Step
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ANNOTATOR, SUBJ_INDEX, TESTSET
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI, SUBJ_INDEX, DOC_INDEX
from eleet_pretrain.datasets.pretraining.mongo_processing.mongo_store import MongoStore, multiprocessing_get_ranges, \
    multiprocessing_mongo_rename, multiprocessing_update_process_bar

from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI, DOC_URI, SUBJ_INDEX, DOC_INDEX, \
    ATTR_DESCRIPTION, ATTR_LABEL, DOC_NUMBER, SUBJECT_URI, HINT_OBJ, HINT_PRED_LABEL, HINT_PRED_URI, ANSWER_START, \
    ANSWER_END, ANSWER_URI, ANSWER_SURFACEFORM, ANSWER_NORMALIZED

logger = logging.getLogger(__name__)

QUALIFIER_ALIGNER = "Qualifier-Aligner"
SPO_ALIGNER = "SPOAligner"
NO_SUBJECT_ALIGNER = "NoSubject-Triple-aligner"
ALL_ENT_ALIGNER = "Simple-Aligner"

FILTER_OUT_ATTR_ALIGNER_PAIRS = [
    [f"http://www.wikidata.org/prop/direct/{predicate}", aligner]
    for predicate, aligners in (
        ("P47", (NO_SUBJECT_ALIGNER, ALL_ENT_ALIGNER)),  # shares border with
        ("P36", (ALL_ENT_ALIGNER, )),  # capital
        ("P530", (NO_SUBJECT_ALIGNER, ALL_ENT_ALIGNER)),  # diplomatic relations
        ("P1589", (NO_SUBJECT_ALIGNER, ALL_ENT_ALIGNER)),  # lowest point
        ("P1365", (NO_SUBJECT_ALIGNER, ALL_ENT_ALIGNER)),  # replaces
        ("P1366", (NO_SUBJECT_ALIGNER, ALL_ENT_ALIGNER)),  # replaced by
        ("P155", (NO_SUBJECT_ALIGNER, ALL_ENT_ALIGNER)),  # follows
        ("P156", (NO_SUBJECT_ALIGNER, ALL_ENT_ALIGNER))  # followed_by
    )
    for aligner in aligners
]

TMP_SPLIT_COLLECTIONS = ["tmp-split-1", "tmp-split-2"]
TMP_QUERY_PREPARATION_COLLECTION = "tmp-query-prep"
TMP_QUERY_PREPARATION_COLLECTIONS = [f"tmp-query-prep-{i}" for i in range(6)]
ATTR_COLS = [ATTR_LABEL, ATTR_URI, ATTR_DESCRIPTION]
DOC_COLS = [DOC_URI, DOC_NUMBER, DOC_INDEX]
SUBJECT_COLS = [SUBJECT_URI, SUBJ_INDEX, TESTSET]
HINT_COLS = [HINT_OBJ, HINT_PRED_URI, HINT_PRED_LABEL]
UNSEEN_SPLIT_SIZE = 0.03
DEVELOP_SPLIT_SIZE = 0.03


class QueryPreparation(BaseLoader):
    """Merge data from the different sources."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_queries(self, num_workers):
        p1 = multiprocessing.Process(target=self.compute_splits, daemon=True)
        p2 = multiprocessing.Process(target=self.group_answer_mentions, daemon=True)
        for f in [p1.start, p2.start, p1.join, p2.join]:
            f()

        logger.info("Prepare and split queries")
        tmp_store = MongoStore(TMP_QUERY_PREPARATION_COLLECTION)
        self.multiprocessing_preprocess(
            data_loader_process=multiprocessing_get_ranges(tmp_store, slice_size=10_000),
            writer_process=multiprocessing_update_process_bar(tmp_store, desc="Split queries"),
            worker_process=self.split_queries,
            num_workers=num_workers,
        )
        multiprocessing_mongo_rename(self.prepared_query_stores, TMP_QUERY_PREPARATION_COLLECTIONS)

        logger.info("Group queries")
        self.multiprocessing_preprocess(
            data_loader_process=multiprocessing_get_ranges(self.prepared_query_stores, slice_size=10_000),
            writer_process=multiprocessing_update_process_bar(self.prepared_query_stores, desc="Group queries"),
            worker_process=self.group_docs,
            num_workers=num_workers,
        )
        multiprocessing_mongo_rename(self.prepared_query_stores, TMP_QUERY_PREPARATION_COLLECTIONS)

        procs = [multiprocessing.Process(target=self.group_queries, daemon=True, args=(i, ))
                 for i in range(len(self.prepared_query_stores))]
        for f in [p.start for p in procs] + [p.join for p in procs]:
            f()

        p = multiprocessing.Process(target=self.wrap_up, daemon=True)
        p.start()
        p.join()

    def wrap_up(self):
        self.mongo_connect()
        for s in self.prepared_query_stores:
            s.create_indexes()
        s = MongoStore(TMP_SPLIT_COLLECTIONS[0])
        s.connect()
        s.drop()
        s = MongoStore(TMP_SPLIT_COLLECTIONS[1])
        s.connect()
        s.drop()
        s = MongoStore(TMP_QUERY_PREPARATION_COLLECTION)
        s.connect()
        s.drop()

    def compute_splits(self):
        self.mongo_connect()
        logger.info("Compute splits")
        tmp_store = MongoStore(TMP_SPLIT_COLLECTIONS[0])
        tmp_store.connect()
        tmp_store.collection.insert_many(
            [{"_id": {"_attr_uri": "http://www.wikidata.org/prop/direct/" + q}} for q in UNSEEN_QUERY_ATTRS])
        tmp_store.create_index(f"_id.{ATTR_URI}")

        self.query_store.aggregate([  # specify doc, attr pairs in develop split
            {"$project": {ATTR_URI: 1, DOC_URI: 1}},
            {"$match": {"$expr": {"$ne": [f"${ATTR_URI}", "id"]}}},
            {"$group": {"_id": {ATTR_URI: f"${ATTR_URI}", DOC_URI: f"${DOC_URI}"}}},
            {"$lookup": {
                "from": TMP_SPLIT_COLLECTIONS[0],
                "localField": f"_id.{ATTR_URI}",
                "foreignField": f"_id.{ATTR_URI}",
                "as": "match"}},
            {"$match": {"match.0": {"$exists": False}}},
            {"$project": {"_split": {"$rand": {}}}},
            {"$match": {"_split": {"$lt": DEVELOP_SPLIT_SIZE}}},
            {"$out": TMP_SPLIT_COLLECTIONS[1]}
        ], allowDiskUse=True)

        tmp_store = MongoStore(TMP_SPLIT_COLLECTIONS[1])
        tmp_store.connect()
        tmp_store.create_index([(f"_id.{ATTR_URI}", 1), (f"_id.{DOC_URI}", 1)])

        self.wikidata_store.create_indexes()  # entities in test splits: skyscraper, countries, nobel
        for i, (split_name, superclass, wikidata_field, pipeline_def) in enumerate(TEST_SPLIT_DEFINITIONS):
            self.wikidata_class_hierarchy_store.aggregate([
                {"$match": {"transitive_superclasses": superclass}},
                {"$project": {"wikidata_id": 1}},
                {"$lookup": {
                    "from": "wikidata",
                    "localField": "wikidata_id",
                    "foreignField": wikidata_field,
                    **pipeline_def,
                    "as": "x"}},
                {"$unwind": "$x"},
                {"$replaceRoot": {"newRoot": "$x"}},
                {"$project": {"_id": 0, "wikidata_id": 1}},
                {"$group": {"_id": "$wikidata_id"}},
                {"$out": f"split-{split_name}"}
            ])
            tmp_split_store = MongoStore(f"split-{split_name}")
            tmp_split_store.connect()
            tmp_split_store.aggregate([{("$merge" if i > 0 else "$out"): SPLIT_TEST_ALL}])

    def group_answer_mentions(self):
        self.mongo_connect()
        self.query_store.aggregate([
            {"$match": {
                "$expr": {"$not": {"$in": [["$_attr_uri", "$_annotator"], FILTER_OUT_ATTR_ALIGNER_PAIRS]}}
            }},
            {"$group": {  # group all the answers for the same queries together
                    "_id": {x: f"${x}" for x in ATTR_COLS + DOC_COLS + SUBJECT_COLS + HINT_COLS
                            + [ANSWER_URI, ANSWER_NORMALIZED]},
                    "_answer_mentions": {"$addToSet": {x: f"${x}"
                                                       for x in [ANSWER_START, ANSWER_END, ANSWER_SURFACEFORM]}},
                    "_annotators": {"$addToSet": f"${ANNOTATOR}"}
            }},
            {"$out": TMP_QUERY_PREPARATION_COLLECTION}
        ], allowDiskUse=True)

    def group_docs(self, job_queue, example_queue, worker_id):
        """
        In a text about entity A and a query with subject A and objects B, C ..., add all queries with subject B, C, ...
        These can later be used to construct multi-row, multi-attribute joins.
        """
        self.mongo_connect()

        job = job_queue.get()
        while job is not None:
            skip, limit, total, split = job

            self.prepared_query_stores[split].create_index("_id._doc")
            self.prepared_query_stores[split].aggregate([  # TODO DOC Nr and Hint
                {"$skip": skip}, {"$limit": limit},
                {"$match": {"$expr": {"$eq": ["$_id._doc", "$_id._subj"]}}},
                {"$lookup": {
                    "from": self.prepared_query_stores[split].collection_name,
                    "localField": "_id._doc",
                    "foreignField": "_id._doc",
                    "as": "_dependent_queries",
                    "let": {"answer": "$_id._answer_uri", "subject": "$_id._subject_uri"},
                    "pipeline": [
                        {"$match": {"$expr": {"$eq": ["$_id._subject_uri", "$$answer"]}}},
                        {"$match": {"$expr": {"$ne": ["$_id._attr_uri", "id"]}}},
                        {"$project": {"_answer_mentions": 1,
                                      "_cycle_back": {"$eq": ["$_id._answer_uri", "$$subject"]},
                                      **{c: f"$_id.{c}" for c in ATTR_COLS + [ANSWER_URI, ANSWER_NORMALIZED]}}},
                        {"$group": {  # group all the answers for the same queries together
                            "_id": {x: f"${x}" for x in ATTR_COLS},
                            "_answers": {"$push": {ANSWER_URI: f"${ANSWER_URI}",
                                                   ANSWER_NORMALIZED: f"${ANSWER_NORMALIZED}",
                                                   "_answer_mentions": "$_answer_mentions"}},
                            "_cycled_back": {"$max": "$_cycle_back"}
                        }},
                        {"$match": {"_cycled_back": False}}, # avoid cycling back
                        {"$unset": "_cycled_back"}
                    ]
                }},
                {"$merge": TMP_QUERY_PREPARATION_COLLECTIONS[split]}
            ])

            example_queue.put((limit, total, split))
            job = job_queue.get()

    def split_queries(self, job_queue, example_queue, worker_id):
        self.mongo_connect()
        tmp_store = MongoStore(TMP_QUERY_PREPARATION_COLLECTION)
        tmp_store.connect()

        job = job_queue.get()
        while job is not None:
            skip, limit, total = job

            logger.info("Preparing Queries")
            tmp_store.aggregate([
                {"$skip": skip}, {"$limit": limit},
                {"$set": {"_attr": {"$last": { "$split": [ f"$_id.{ATTR_URI}", "/" ] }}}},
                {"$lookup": {
                    "from": self.wikidata_store.collection_name,  # merge description
                    "localField": "_attr",
                    "foreignField": "wikidata_id",
                    "pipeline": [
                        {"$project": {"description": 1}}
                    ],
                    "as": "_desc"
                }},
                {"$set": {f"_id.{ATTR_DESCRIPTION}": {"$first": "$_desc.description"}}}, {"$unset": ["_desc", "_attr"]},
                {"$lookup": {
                    "from": TMP_SPLIT_COLLECTIONS[0],
                    "localField": f"_id.{ATTR_URI}",
                    "foreignField": f"_id.{ATTR_URI}",
                    "as": "_split1"
                }},
                {"$lookup": {
                    "from": TMP_SPLIT_COLLECTIONS[1],
                    "localField": f"_id.{ATTR_URI}",
                    "foreignField": f"_id.{ATTR_URI}",
                    "let": {"d": f"$_id.{DOC_URI}"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {"$eq": [ f"$_id.{DOC_URI}",  f"$$d" ]}
                            }
                        }
                    ],
                    "as": "_split2"
                }},
                *[{"$lookup": {
                    "from": f"split-{s}",
                    "localField": f"_id.{DOC_INDEX}",
                    "foreignField": "_id",
                    "as": s

                }} for s, _, _, _ in TEST_SPLIT_DEFINITIONS
                ],
                {"$project": {
                    "_answer_mentions": 1,
                    "_split": {
                        "$switch": {
                            "branches": [
                                *[{
                                    "case": { "$gt" : [ {"$size": f"${s}"}, 0 ] },
                                    "then": 3 + si
                                } for si, (s, _, _, _) in enumerate(TEST_SPLIT_DEFINITIONS)],
                                {
                                    "case": { "$gt" : [ {"$size": "$_split1"}, 0 ]},
                                    "then": 1
                                },
                                {
                                    "case": { "$gt" : [ {"$size": "$_split2"}, 0 ] },
                                    "then": 2
                                },
                            ], "default": 0
                        }
                    },
                }}, 
                {"$out": f"query-tmp-{worker_id}"}
            ])
            tmp = MongoStore(f"query-tmp-{worker_id}")
            tmp.connect()
            for i, store_name in enumerate(TMP_QUERY_PREPARATION_COLLECTIONS):
                tmp.aggregate([{"$match": {"$or": [{"_split": i}, {"_id._attr_uri": "id"}]}}, {"$merge": store_name}])
            tmp.drop()
            example_queue.put((limit, total))
            job = job_queue.get()


    def group_queries(self, split):               
        self.mongo_connect()
        self.prepared_query_stores[split].aggregate([
            {
                "$group": {  # Group different answers for same query
                    "_id": {**{x: f"$_id.{x}" for x in DOC_COLS + ATTR_COLS}},
                    "_answers": {"$push": {ANSWER_URI: f"$_id.{ANSWER_URI}",
                                           ANSWER_NORMALIZED: f"$_id.{ANSWER_NORMALIZED}",
                                           "_answer_mentions": "$_answer_mentions",
                                           "_dependent_queries": "$_dependent_queries"}},
                }
            },
            {
                "$group": {  # Group different query attributes together
                    "_id": {**{x: f"$_id.{x}" for x in DOC_COLS}},
                    "_queries": {"$push": {"_answers": "$_answers", **{x: f"$_id.{x}" for x in ATTR_COLS}}}
                }
            },
            {"$match": {"$or": [{"_queries.1": {"$exists": 1}}, {"_queries._attr_uri": {"$ne": "id"}}]}},
            {"$out": self.prepared_query_stores[split].collection_name}
        ], allowDiskUse=True)
class QueryPreparationStep(Step):
    """Load data from wikidata dump."""

    depends_on = {TRExPreprocessStep}

    def check_done(self, args, dataset):
        """Check whether the step has already been executed."""
        x = QueryPreparation(dataset, args.dataset_dir, args.small_sample)
        x.mongo_connect()
        return not any(y.is_empty() for y in x.prepared_query_stores)

    def run(self, args, dataset):
        """Execute the step."""
        x = QueryPreparation(dataset, args.dataset_dir, args.small_sample)
        x.prepare_queries(args.num_workers)
