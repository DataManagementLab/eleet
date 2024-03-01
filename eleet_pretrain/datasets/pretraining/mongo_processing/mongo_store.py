from contextlib import ExitStack
import os
from pymongo import MongoClient
import logging
import multiprocessing

from tqdm import tqdm
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI, DOC_INDEX,  SUBJ_INDEX

logger = logging.getLogger(__name__)

def check_connected(f):
    def func(self, *args, **kwargs):
        if not self.connected:
            msg = "Not connected to mongodb instance!"
            logger.warn(msg)
            # print(msg)
            return None
        return f(self, *args, **kwargs)
    return func

def _clean(*cn):
    for c in cn:
        s = MongoStore(c)
        s.connect()
        s.drop()

def multiprocessing_mongo_rename(stores, tmp_store_names):
    def _rename(i):
        nonlocal stores, tmp_store_names
        sink, tmp = stores[i], tmp_store_names[i]
        s = MongoStore(tmp)
        s.connect()
        s.rename(sink.collection_name, dropTarget=True)
        sink.connect()
        sink.create_indexes()
        _clean(tmp)

    p = [0] * len(stores)
    for i in range(len(stores)):
        p[i] = multiprocessing.Process(target=_rename, daemon=True, args=(i, ))
        p[i].start()
    for i in range(len(stores)):
            p[i].join()


def multiprocessing_mongo_clean(self, *collection_name):
    p = multiprocessing.Process(target=_clean, daemon=True, args=tuple(collection_name))
    p.start()
    p.join()


def multiprocessing_get_ranges(mongo_store_s, slice_size):
    """Get offset and limit for each proceessing job on mongodb-store."""
    def f(job_queue, num_workers):
        nonlocal mongo_store_s, slice_size
        put_split = True
        if not isinstance(mongo_store_s, list):
            put_split = False
            mongo_store_s = [mongo_store_s]
        for s in mongo_store_s:
            s.connect()
        for split in range(len(mongo_store_s)):
            count = mongo_store_s[split].estimated_document_count()
            for i in range(0, count + slice_size, slice_size):
                job_queue.put((i, slice_size, count, split) if put_split else (i, slice_size, count))
        for _ in range(num_workers):
            job_queue.put(None)
    return f


def multiprocessing_update_process_bar(mongo_store_s, desc, sample_file=False, loader=None, position=0):
    """Update the process bar for a mongodb multiprocessing task."""
    def f(example_queue):
        nonlocal mongo_store_s, desc, sample_file, loader
        put_split = True
        sample_file_data = dict()
        if not isinstance(mongo_store_s, list):
            put_split = False
            mongo_store_s = [mongo_store_s]
        pbars = {}
        with ExitStack() as stack:
            for split in range(len(mongo_store_s)):
                pbars[split] = stack.enter_context(tqdm(desc=f"{desc} {split}" if put_split else desc,
                                                        position=position + split))
            job = example_queue.get()
            while job is not None:
                if not put_split:
                    job = job + (0, )
                if not sample_file:
                    job = job + (0, )
                num, total, split, sfd = job
                pbars[split].total = total
                pbars[split].update(num)
                job = example_queue.get()
                if sample_file:
                    sample_file_data = loader.merge_sample_file_data(sample_file_data, sfd)
        if sample_file:
            loader.generate_sample_files(sample_file_data)
    return f


class MongoStore():

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.connected = False


    def connect(self, user=None, password=None, host=None, port=None, db=None):
        if self.connected:
            return
        user = user or os.environ["MONGO_USER"]
        password = password or os.environ["MONGO_PASSWORD"]
        host = host or os.environ["MONGO_HOST"]
        port = int(port or os.environ["MONGO_PORT"])
        db = db or os.environ["MONGO_DB"]
        self.client = MongoClient(host, port, username=user, password=password, authSource="admin")
        self.db = self.client.get_database(db)
        self.collection = self.db.get_collection(self.collection_name)
        self.connected = True

    @check_connected
    def insert_many(self, documents, *args, **kwargs):
        self.collection.insert_many(documents, *args, **kwargs)

    @check_connected
    def delete_many(self, filter):
        self.collection.delete_many(filter)

    @check_connected
    def is_empty(self):
        return self.collection.estimated_document_count() == 0

    @check_connected
    def __iter__(self):
        return self.collection.find({})

    @check_connected
    def __getattr__(self, key):
        return getattr(self.collection, key)


class KelmStore(MongoStore):
    def __init__(self):
        super().__init__("kelm")

    @check_connected
    def __getitem__(self, key):
        return self.collection.find({"subject_id": key})

    @check_connected
    def subject_ids(self):
        return self.collection.distinct("subject_id")


class WikidataStore(MongoStore):
    def __init__(self):
        super().__init__("wikidata")

    @check_connected
    def __getitem__(self, key):
        return self.collection.find_one({"wikidata_id": key})

    @check_connected
    def create_indexes(self):
        logger.info("Creating Indexes for wikidata store")
        self.collection.create_index("wikidata_id")
        self.collection.create_index("superclasses")
        self.collection.create_index("types")
        self.collection.create_index("neighbors.o")


class WikidataClassHierarchyStore(MongoStore):
    def __init__(self):
        super().__init__("wikidata-class-hierarchy")

    @check_connected
    def __getitem__(self, key):
        return self.collection.find_one({"wikidata_id": key})

    @check_connected
    def create_indexes(self):
        logger.info("Creating Indexes for wikidata class hierarchy store")
        self.collection.create_index("wikidata_id")


class QueriesStore(MongoStore):
    def __init__(self, dataset_name):
        super().__init__(f"{dataset_name}-queries")

    def create_indexes(self):
        logger.info("Create indexes for queries store.")
        self.collection.create_index(DOC_INDEX)
        self.collection.create_index(SUBJ_INDEX)
        self.collection.create_index(ATTR_URI)


class TextStore(MongoStore):
    def __init__(self, dataset_name):
        super().__init__(f"{dataset_name}-text")

    def create_indexes(self):
        logger.info("Create indexes for text store.")
        self.collection.create_index(DOC_INDEX)


class PreparedTextStore(MongoStore):
    def __init__(self, dataset_name):
        super().__init__(f"{dataset_name}-prepared-text")

    def create_indexes(self):
        logger.info("Create indexes for prepared-text store.")
        self.collection.create_index("text_idx")


class PreparedQueryStore(MongoStore):
    def __init__(self, dataset_name, split):
        super().__init__(f"{dataset_name}-prepared-query-{split}")
        self.split = split

    def create_indexes(self):
        logger.info("Create indexes for prepared-query store.")
        self.collection.create_index("_groupby_bucket")
        self.collection.create_index("_queries.0._answers.0._answer_uri")
        self.collection.create_index("_queries._attr_uri")
        self.collection.create_index("_evidence_types")
