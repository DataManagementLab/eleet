"""Input Formatter for multi-modal DBs with TaBERT."""

import multiprocessing
import queue
import uuid
import argparse
import cProfile
import logging
import re
import os
from collections import namedtuple

import tempfile
import h5py
from pympler import summary, muppy, asizeof
from pathlib import Path
import spacy
from tqdm import tqdm
import transformers
from eleet_pretrain.datasets import DataCombineStep
from transformers import BertTokenizerFast
from eleet_pretrain.datasets.base_loader import AccessRecordingDict, BaseLoader, ParquetStore
from eleet_pretrain.datasets.pretraining.mongo_processing.mongo_store import MongoStore
from eleet_pretrain.model.config import VerticalEleetConfig
from eleet_pretrain.utils import insert_into_global_log_begin, insert_into_global_log_end
from eleet_pretrain.steps import Step
from eleet_pretrain.utils import logging_setup
from eleet_pretrain.datasets.input_formatting.tabert_input_formatter import EleetTaBertInputFormatter
from eleet_pretrain.datasets.input_formatting.base_input_formatter import BaseEleetInputFormatter

logger = logging.getLogger(__name__)

InputFormattingSetting = namedtuple("InputFormattingSetting", ("formatter_cls", "config", "suffix"))

SETTINGS = [
        InputFormattingSetting(EleetTaBertInputFormatter, VerticalEleetConfig, None)
]

def profile_this(func):
    def f(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        func(*args, **kwargs)
        profiler.disable()
        profiler.dump_stats(f"{func.__name__}-{uuid.uuid4().hex}.prof")
    return f


class EmptyJobException(ValueError):
    pass

class EleetInputTensoriser(BaseLoader):

    def run_input_formatter(self, num_workers, model_name_or_path, evidence_mode, args):
        """Run the input formatter for the given split."""
        available_splits = sorted(set(map(str, self.store.nodes())) - {"train_default"})
        available_splits = ["train_default"] + available_splits

        if args.only_formatting_of_dataset:
            available_splits = args.only_formatting_of_dataset

        if args.skip_formatting_of_dataset:
            regex = re.compile("(" + "|".join(args.skip_formatting_of_dataset) + ")")
            skipped = set(x for x in available_splits if regex.match(x))
            available_splits = [x for x in available_splits if x not in skipped]
            logger.info(f"Skipping datasets: {skipped}")

        for setting in SETTINGS:
            config = setting.config(base_model_name=model_name_or_path)
            store_destination = BaseLoader.get_final_preprocessed_data_path(self.dataset_dir, self.sample,
                                                                            self.dataset, args, setting.suffix)
            h5py.File(store_destination, "w")  # creates the file
            for split in available_splits:
                with tempfile.TemporaryDirectory(dir=store_destination.parent) as tempdir:
                    self.multiprocessing_preprocess(
                        data_loader_process=self.get_nodes,
                        writer_process=self.write,
                        worker_process=self.format_batch,
                        num_workers=num_workers,
                        write_queue_maxsize=100,
                        writer_args=(split, store_destination),
                        loader_args=(split,),
                        worker_args=(setting, config, tempdir, evidence_mode, args.skip_formatting_of_table_name),
                        num_error_retries=0 if self.sample else 20,
                        job_limit_worker=5
                    )

    # @profile_this
    def get_nodes(self, split, job_queue, num_workers):
        logger.info(f"Run input formatting on {split} split.")
        for data in self.store.get_node(split).nodes():
            job_queue.put(data)
            # mem("get_nodes")
        for _ in range(num_workers):
            job_queue.put(None)

    # @profile_this
    def write(self, split, store_destination, example_queue):
        with tqdm(desc=f"Run input formatting on {split} split") as pbar:
            job = example_queue.get()
            pbar.total = len(self.store.get_node(split).nodes())
            while job is not None:
                pbar.update(1)
                BaseLoader.merge_final_results(
                    store_destination=store_destination,
                    partial_result_path=job,
                    split=split,
                    write_mode="a"
                )
                job = example_queue.get()
                # mem("write")

    # @profile_this
    def format_batch(self, setting, config, tempdir, evidence_mode, skip_table_name,
                    job_queue, example_queue, worker_id, q=queue.Queue(), max_num_jobs=float("inf")):
        tokenizer = BertTokenizerFast.from_pretrained(config.base_model_name)
        input_formatter = setting.formatter_cls(config, tokenizer)
        i = 0
        while i < max_num_jobs:
            job = job_queue.get()
            if job is None:
                break
            path = Path(tempdir) / f"{worker_id}-{i}.h5"
            try:
                self.run_format_job(job, input_formatter, path, evidence_mode, skip_table_name)
                # mem("format_batch", worker_id, i)
            except EmptyJobException:
                continue
            example_queue.put(path)
            i += 1
        q.put(job is None)

    def run_format_job(self, data, input_formatter, path, evidence_mode, skip_table_name):
        if len(data["queries"]) == 0:
            raise EmptyJobException

        tables, column_ids, queries, texts, answers, overlap_mentions, header_queries, header_meta = self.collect(data)
        instances = input_formatter.get_instances_preprocessing(
            tables=tables,
            column_ids=column_ids,
            queries=queries,
            answers=answers,
            mentions=overlap_mentions,
            texts=texts,
            header_queries=header_queries,
            header_meta=header_meta,
            evidence_mode=evidence_mode,
            skip_table_re=re.compile("(" + "|".join(skip_table_name) + ")") if skip_table_name else None
        )
        tensor_dict = input_formatter.get_tensor_dict(
            instances=instances,
            subdir=data.subdir,
        )
        if len(tensor_dict) == 0:
            raise EmptyJobException
        self.store_final_results(path, split="tmp", encodings=tensor_dict, write_mode="w")

    def collect(self, data):
        tables, column_ids, queries, texts, answers, overlap_mentions, header_queries, header_meta = \
            [], [], [], [], [], [], [], []
        for i, columns in data["header_columns"].iterrows():
            rows = data["rows"].loc[i]
            rows.columns = columns[:len(rows.columns)]
            column_ids.append(data["header_column_ids"].loc[i])
            header_meta.append(data["header_meta"].loc[i])
            tables.append(rows)
            texts.append(data["texts"].loc[i])
            queries.append(data["queries"].loc[i].sort_index())
            answers.append(data["answers"].loc[i].sort_index())
            overlap_mentions.append(data["overlap_mentions"].loc[i])
            if i in data["header_queries"].index.unique(level="table_id"):
                header_queries.append(data["header_queries"].loc[i])
            else:
                header_queries.append(data["header_queries"].loc[[]])
            # if i in data["align_text"].index.unique(level="table_id"):
            #     align_text.append(data["align_text"].loc[i])
            # else:
            #     align_text.append(data["align_text"].loc[[]])
        return tables, column_ids, queries, texts, answers, overlap_mentions, header_queries, header_meta


class InputFormattingStep(Step):
    """Load data from wikidata dump."""
    depends_on = {DataCombineStep}

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Choose tokenizer to use."""
        parser.add_argument("--model-name-or-path", default="bert-base-uncased")
        parser.add_argument("--skip-formatting-of-table-name", nargs="+", type=str, default=[])
        parser.add_argument("--skip-formatting-of-dataset", nargs="+", type=str, default=[])
        parser.add_argument("--only-formatting-of-dataset", nargs="+", type=str, default=None)
        parser.add_argument("--evidence-mode", type=str, choices=["full", "row-id", "none"], default="full")

    def check_done(self, args, dataset):
        p = BaseLoader.get_final_preprocessed_data_path(args.dataset_dir, args.small_sample, dataset, args)
        if not p.exists():
            return False
        with h5py.File(p) as f:
            return len(f) > 0

    def run(self, args, dataset):
        """Execute the step."""
        loader = EleetInputTensoriser(dataset, args.dataset_dir, args.small_sample)
        loader.run_input_formatter(args.num_workers, args.model_name_or_path, args.evidence_mode, args)

    @staticmethod
    def logging_setup(args, dataset):
        global_log = BaseLoader.get_global_preprocessing_log_path(args.dataset_dir)
        logging_path = BaseLoader.get_preprocessing_log_path(args.dataset_dir, args.small_sample, dataset, args)
        output_path = BaseLoader.get_final_preprocessed_data_path(args.dataset_dir, args.small_sample, dataset, args)
        logging_setup(args.log_level, logging_path)
        start_date = insert_into_global_log_begin(global_log, logging_path, output_path=output_path)
        return lambda exception: insert_into_global_log_end(global_log, start_date, exception)


# def mem(*name):
#     o = muppy.get_objects()
#     pid = os.getpid()
#     total = psutil.Process(pid).memory_info().rss / 1024 ** 2
#     with open("mem-profiling", "a") as f:
#         to_check = [BaseLoader, BaseEleetInputFormatter, AccessRecordingDict, ParquetStore, MongoStore,
#                     multiprocessing.queues.Queue, transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
#                     spacy.language.Language]
#         for line in summary.format_(summary.summarize(o)):
#             print(line, file=f)
#         for t in to_check:
#             objs = [x for x in o if hasattr(x, "__class__") and isinstance(x, t)]
#             size = asizeof.asizeof(objs) / 1024 ** 2
#             print(pid, *name, ": Size of", t.__name__, round(size, 3), "(", len(objs), ")", "total:", int(total),
#                   file=f)
