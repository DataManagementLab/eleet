"""Load Trex-dataset into tabular data and text-table."""

import abc
from datetime import datetime
import logging
import shutil
import sys
import numpy as np
from eleet_pretrain.datasets.pretraining.mongo_processing.mongo_store import KelmStore, QueriesStore, PreparedQueryStore, \
    PreparedTextStore, TextStore, WikidataClassHierarchyStore, WikidataStore
import multiprocessing
from pathlib import Path
from collections import namedtuple
from eleet_pretrain.datasets.pretraining.data_import.wikidata_utils import PseudoDict, WikidataProperties

import h5py
import pandas as pd
import ujson as json
from tqdm import tqdm
from eleet_pretrain.utils import count_lines, get_date_prefix, get_git_arg_hashes

logger = logging.getLogger(__name__)

class BaseLoader(abc.ABC):  # pylint: disable=too-many-public-methods
    """Load TREx dataset and put in in the right format."""

    def __init__(self, dataset, dataset_dir, sample):
        """Initialize the loader."""
        cls = type(self)
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.sample = sample
        self.output_dir = cls.get_output_dir(self.dataset_dir)
        self.wikidata_dir = cls.get_wikidata_dir(self.dataset_dir, sample)
        self.trex_dir = cls.get_trex_dir(self.dataset_dir, sample)
        self.kelm_dir = cls.get_kelm_dir(self.dataset_dir, sample)
        self.output_dir.mkdir(exist_ok=True)
        self.store = PreliminaryResultsStore(self.output_dir, dataset)
        self.kelm_store = KelmStore()
        self.wikidata_store = WikidataStore()
        self.wikidata_class_hierarchy_store = WikidataClassHierarchyStore()
        self.query_store = QueriesStore(dataset)
        self.text_store = TextStore(dataset)
        self.prepared_text_store = PreparedTextStore(dataset)
        self.prepared_query_stores = [
            PreparedQueryStore(dataset, 0),
            PreparedQueryStore(dataset, 1),
            PreparedQueryStore(dataset, 2),
            PreparedQueryStore(dataset, 3),
            PreparedQueryStore(dataset, 4),
            PreparedQueryStore(dataset, 5),
        ]
        self._labels = None
        self._aliases = None
        self.labels_path = cls.get_labels_path(self.dataset_dir)
        self.aliases_path = cls.get_aliases_path(self.dataset_dir)
        self._multiprocessing_manager = None
        self.rng = np.random.default_rng(42)

    @property
    def multiprocessing_manager(self):
        if self._multiprocessing_manager is None:
            self._multiprocessing_manager = multiprocessing.Manager()
        return self._multiprocessing_manager

    def mongo_connect(self):
        self.kelm_store.connect()
        self.wikidata_store.connect()
        self.wikidata_class_hierarchy_store.connect()
        self.query_store.connect()
        self.text_store.connect()
        self.prepared_text_store.connect()
        for s in self.prepared_query_stores:
            s.connect()

    def _dict_from_wikidata(self, which_dict, transform_func=lambda x: x):
        self.mongo_connect()
        def f(key):
            entity = self.wikidata_store[key]
            if entity is None:
                return {}
            if which_dict not in self.wikidata_store[key]:
                return None
            return transform_func(self.wikidata_store[key][which_dict])
        return PseudoDict(f)

    @property
    def labels(self):
        """Return mapping from Wikidata-Entity to its labels."""
        if self.labels_path.exists():
            return self._autoload("labels")
        return self._dict_from_wikidata("labels")

    @property
    def aliases(self):
        """Return mapping from Wikidata-Entity to its aliases."""
        if self.aliases_path.exists():
            return self._autoload("aliases")
        return self._dict_from_wikidata("aliases")

    @property
    def types(self):
        """Return mapping from Wikidata-Entity to its types."""
        return self._dict_from_wikidata("types")

    @property
    def superclasses(self):
        """Return mapping from Wikidata-Entity to its superclasses."""
        return self._dict_from_wikidata("superclasses")

    @property
    def properties(self):
        """Return statement qualifiers."""
        return self._dict_from_wikidata("properties", lambda x: WikidataProperties(x, self.labels, self.aliases,
                                                                                   self.rng))

    @classmethod
    def get_labels_path(cls, dataset_dir, sample=False):
        """Get the path to store labels."""
        return cls.prepend_sample(sample, BaseLoader.get_output_dir(dataset_dir) / "labels", True)

    @classmethod
    def get_aliases_path(cls, dataset_dir, sample=False):
        """Get the path to Wikidata aliases."""
        return cls.prepend_sample(sample, BaseLoader.get_output_dir(dataset_dir) / "aliases", True)

    def get_sample_file_data(self, *names):
        result = dict()
        for name in names:
            d = getattr(self, "_" + name)
            if not self.sample or not isinstance(d, AccessRecordingDict):
                continue
            result[name] = d.get_subdict()
            d.accessed_keys = set()
        return result
    
    def merge_sample_file_data(self, *dicts):
        result = dict()
        for d in dicts:
            for name in d:
                if name not in result:
                    result[name] = dict()
                result[name].update(d[name])
        return result

    def generate_sample_files(self, sample_data):
        for name in sample_data:
            sample_path = getattr(type(self), f"get_{name}_path")(self.dataset_dir, self.sample)

            with open(sample_path, "w") as f:
                json.dump(sample_data[name], f)


    def _autoload(self, name):
        """Autoload member when accessed."""
        if getattr(self, "_" + name) is not None:
            return getattr(self, "_" + name)

        path = getattr(type(self), f"get_{name}_path")(self.dataset_dir, self.sample)
        if (
            self.sample and  # noqa
            not path.exists()
        ):
            d = AccessRecordingDict()
        else:
            d = self.multiprocessing_manager.dict()
        if self.sample and path.exists():
            setattr(self, name + "_path", path)
        getattr(self, "_load_" + name)(d)
        return getattr(self, "_" + name)

    def _load_labels(self, dictionary):
        """Load cached labels from disk."""
        self._labels = BaseLoader._load_jsonl_update(self.labels_path, "labels", dictionary)

    def _load_aliases(self, dictionary):
        """Load cached aliases from disk."""
        self._aliases = BaseLoader._load_jsonl_update(self.aliases_path, "aliases", dictionary)

    @staticmethod
    def _load_jsonl_update(path, name, dictionary):
        """Load a jsonl file with a dictionary in each line."""
        with open(path) as f:
            for line in tqdm(f, desc=f"Loading {name} from disk", total=count_lines(path)):
                dictionary.update(json.loads(line))
        return dictionary



    @staticmethod
    def store_final_results(store_destination: Path, split, encodings, write_mode):
        """Store final results."""
        store_destination.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(store_destination, write_mode) as file:
            group = file.require_group(split)
            for key, value in encodings.items():
                if key in ("sample_size",):
                    continue
                logger.info(
                    f"Stored {key}-tensor of shape {value.shape} for {split} split in {store_destination.absolute()}")
                logger.debug(f"Tensor:\n{value}")

                if key in group:
                    dataset = group[key]
                    dataset.resize(value.shape[0] + dataset.shape[0], axis=0)
                    dataset[-value.shape[0]:] = value
                else:
                    group.create_dataset(key, maxshape=(None, *value.shape[1:]), data=value, compression="lzf")

    @staticmethod
    def merge_final_results(store_destination: Path, partial_result_path: Path, split, write_mode):
        with h5py.File(partial_result_path, "r") as file:
            BaseLoader.store_final_results(store_destination=store_destination,
                                           split=split,
                                           encodings={k: file["tmp"][k] for k in file["tmp"]},
                                           write_mode=write_mode)
        partial_result_path.unlink()

    @staticmethod
    def multiprocessing_preprocess(data_loader_process, writer_process, worker_process, num_workers,
                                   worker_args=(), loader_args=(), writer_args=(), job_queue_maxsize=1000,
                                   write_queue_maxsize=1000, num_error_retries=20, job_limit_worker=None):
        """Do preprocessing in a multiprocessing fashion."""
        job_queue = multiprocessing.Queue(maxsize=job_queue_maxsize)
        example_queue = multiprocessing.Queue(maxsize=write_queue_maxsize)
        start_worker_args = dict(
            worker_process=worker_process, worker_args=worker_args,
            num_error_retries=num_error_retries, job_limit_worker=job_limit_worker,
            job_queue=job_queue, example_queue=example_queue)

        loader = multiprocessing.Process(target=data_loader_process, daemon=True,
                                         args=(*loader_args, job_queue, num_workers))
        loader.start()

        workers = []
        for i in range(num_workers):
            worker, communication_queue = BaseLoader.multiprocessing_start_worker(**start_worker_args, worker_id=i)
            workers.append((worker, communication_queue))

        writer = multiprocessing.Process(target=writer_process, daemon=True, args=(*writer_args, example_queue, ))
        writer.start()

        BaseLoader.join_workers(workers, job_limit_worker, start_worker_args)
        loader.join()

        example_queue.put(None)
        writer.join()
        example_queue.close()
        job_queue.close()

    @staticmethod
    def join_workers(workers, job_limit_worker, start_worker_args):
        if not job_limit_worker:
            for worker, _ in workers:
                worker.join()
            logger.info("Workers done")
            print("Workers done")
            return

        i = 0
        worker_id = len(workers)
        while workers:
            i %= len(workers)
            worker, communication_queue = workers[i]
            worker.join(1)  # (60 * 5)
            if not communication_queue.empty():
                done = communication_queue.get()
                worker.join()
                if done:
                    del workers[i]
                    continue

                new_worker, new_communication_queue = BaseLoader.multiprocessing_start_worker(
                    **start_worker_args, worker_id=worker_id)
                workers[i] = (new_worker, new_communication_queue)
                worker_id += 1

            i += 1

    @staticmethod
    def multiprocessing_start_worker(worker_process, worker_args, num_error_retries, job_limit_worker, job_queue,
                                     example_queue, worker_id):
        logger.info(f"Start worker {worker_id}")
        communication_queue = None
        job_limit_args = ()
        if job_limit_worker: 
            communication_queue = multiprocessing.Queue(maxsize=1)
            job_limit_args = communication_queue, job_limit_worker

        args = (*worker_args, job_queue, example_queue, worker_id, *job_limit_args)
        if isinstance(worker_process, multiprocessing.Process):
            worker = worker_process(*args, daemon=True)
        else:
            worker = multiprocessing.Process(target=BaseLoader.multiprocessing_log_errors(worker_process,
                                                                                          num_error_retries),
                                                 daemon=True, args=args)
        worker.start()
        return worker, communication_queue
    
    @staticmethod
    def multiprocessing_log_errors(func, num_retries):
        def f(*args, **kwargs):
            for _ in range(num_retries + 1):
                try:
                    func(*args, **kwargs)
                    break
                except KeyboardInterrupt as e:
                    logger.info(f"Shutting Down: {e}", exc_info=True)
                    print(type(e), e)
                    sys.stdout.flush()
                    break
                except Exception as e:
                    print(type(e), e)
                    sys.stdout.flush()
                    logger.warn(str(e), exc_info=True)
                    if num_retries == 0:
                        raise e
            else:
                print("Too many errors. Worker shutting down.")
                sys.stdout.flush()
                logger.error("Too many errors. Worker shutting down.")
        return f

    @staticmethod
    def get_output_dir(dataset_dir):
        """Get the path to store intermediate outputs."""
        return dataset_dir / ".output"

    @staticmethod
    def get_final_preprocessed_dir(dataset_dir, sample, dataset_name, args):
        """Get the directory to store the final preprocessed data."""
        git_hash, args_hash = get_git_arg_hashes(args)
        return BaseLoader.prepend_sample(
            sample, dataset_dir / "preprocessed_data" /f"preprocessed_{dataset_name}_{git_hash}_{args_hash}"
        )

    @classmethod
    def prepend_sample(cls, is_sample, path, more_info_if_sample=False):
        """Prepend prefix for small sample files."""
        path_name = path.name
        if is_sample and more_info_if_sample:
            path_name = cls.__name__ + "." + path_name
        if is_sample:
            return path.parent / ("sample." + path_name)
        return path

    @staticmethod
    def get_wikidata_dir(dataset_dir, sample):  # pylint: disable=unused-argument
        """Get the path where wikidata dump is stored."""
        return dataset_dir / "Wikidata"

    @staticmethod
    def get_trex_dir(dataset_dir, sample):
        """Get the path where TREx dataset is stored."""
        return BaseLoader.prepend_sample(sample, dataset_dir / "TREx")

    @staticmethod
    def get_kelm_dir(dataset_dir, sample):
        """Get the path where KELM corpus is stored."""
        return dataset_dir / "KELM"

    @staticmethod
    def get_wikipedia_dir(dataset_dir, sample):
        """Get the path where the Wikipedia dataset is stored."""
        return BaseLoader.prepend_sample(sample, dataset_dir / "Wikipedia")

    @staticmethod
    def get_webtables_dir(dataset_dir, sample):
        """Get the path where the Webtables dataset is stored."""
        return BaseLoader.prepend_sample(sample, dataset_dir / "Webtables")

    @staticmethod
    def get_final_preprocessed_data_path(dataset_dir, sample, dataset_name, args, suffix=""):
        """Get the path where TREx dataset is stored."""
        suffix = suffix or ""
        return BaseLoader.get_final_preprocessed_dir(dataset_dir, sample, dataset_name, args) / f"data{suffix}.h5"

    @staticmethod
    def get_preprocessing_log_path(dataset_dir, sample, dataset_name, args):
        """Get the path where TREx dataset is stored."""
        filename = "_".join([get_date_prefix(), "preprocessing.log"])
        return BaseLoader.get_final_preprocessed_dir(dataset_dir, sample, dataset_name, args) / filename

    @staticmethod
    def get_global_preprocessing_log_path(dataset_dir):
        """Get the global log file, where each start of a preprocessing run is logged."""
        return dataset_dir / "preprocessing.log"


class ParquetStore():
    """A store based on parquet files."""

    def __init__(self, directory):
        """Initialize."""
        self.dir = Path(directory).absolute()
        self.dir.mkdir(exist_ok=True, parents=True)

    def reset(self):
        shutil.rmtree(self.dir)
        self.dir.mkdir(exist_ok=True, parents=True)

    def __getitem__(self, key):
        """Get a DataFrame or Series."""
        path = self.dir / (key.lstrip("/") + ".pq")
        if not path.exists():
            path = self.dir / (key.lstrip("/") + ".pqs")
            return pd.read_parquet(path, pre_buffer=False)["0"]
        return pd.read_parquet(path, pre_buffer=False)

    def __setitem__(self, key, value):
        """Store a DataFrame or Series."""
        suffix = ".pq"
        if isinstance(value, pd.Series):
            suffix = ".pqs"
            value = pd.DataFrame({"0": value})
        path = self.dir / (key.lstrip("/") + suffix)
        path.parent.mkdir(exist_ok=True, parents=True)
        value.to_parquet(path)

    def keys(self):
        """Get all the datasets in the store."""
        return set(str(x.relative_to(self.dir))[:-3] for x in self.dir.rglob("*.pq")) \
            | set(str(x.relative_to(self.dir))[:-4] for x in self.dir.rglob("*.pqs"))

    def direct_keys(self):
        """Get all direct datasets."""
        return set(".".join(str(x.relative_to(self.dir)).split(".")[:-1]) for x in self.dir.iterdir() if x.is_file())


class StoreNodeList(list):
    """A container of store nodes."""
    def __init__(self, iterable):
        super().__init__(sorted(iterable))

    def __contains__(self, key):
        """Containment of strings."""
        return any(key == str(node) for node in self)


class PreliminaryResultsStore(ParquetStore):
    """A store base on parquet to store preliminary results."""

    def __init__(self, directory, dataset_name):
        """Initialize the store."""
        super().__init__(Path(directory) / "preliminary_results" / dataset_name)

    def get_node(self, subdir):
        """Get a particular subnode."""
        return StoreNode(self, subdir)

    def nodes(self):
        """Get all direct subnodes / directories."""
        return StoreNodeList(StoreNode(self, x.relative_to(self.dir)) for x in self.dir.iterdir() if x.is_dir())


class StoreNode(ParquetStore):
    """A node (=directory) in the preliminary result store."""

    def __init__(self, store, subdir):
        """Initialize."""
        self.store = store
        self.subdir = Path(subdir)
        assert (self.store.dir / self.subdir).exists()
        super().__init__(self.store.dir / self.subdir)

    def get_node(self, subdir):
        """Get a particular subnode."""
        return StoreNode(self.store, self.subdir / subdir)

    def nodes(self):
        """Get all direct subnodes / directories."""
        return StoreNodeList(StoreNode(self.store, x.relative_to(self.store.dir))
                             for x in self.dir.iterdir() if x.is_dir())

    def __str__(self):
        """Convert to a string."""
        return self.subdir.name

    def __lt__(self, other):
        return str(self.store.dir / self.subdir) < str(other.store.dir / other.subdir)


class AccessRecordingDict(dict):
    """A dictionary that records which keys are accessed."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.accessed_keys = set()

    def __getitem__(self, key):
        """Get an element."""
        if key in self:
            self.accessed_keys.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        """Get an element."""
        if key in self:
            self.accessed_keys.add(key)
        return super().get(key, default)

    def get_subdict(self):
        """Get dictionary with only accessed keys."""
        return {key: self[key] for key in self.accessed_keys}