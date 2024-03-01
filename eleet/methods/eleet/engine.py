from collections import defaultdict, namedtuple
from datetime import datetime
import enum
from functools import partial
import os
from pathlib import Path
import pickle
import queue
import shutil
from threading import Thread
from time import sleep
import uuid
from attr import Factory
import numpy as np
import pandas as pd
from attrs import define, field
import torch
from eleet_pretrain.model.config import VerticalEleetConfig
from eleet.methods.base_engine import BaseEngine, EngineMode, EngineModeTuple
from eleet.methods.eleet.dataset import ELEETInferenceDataset
from torch.utils.data import DataLoader
from eleet.methods.eleet.model import ELEETInferenceModel
from transformers import PreTrainedTokenizerFast
import torch.multiprocessing as mp
from eleet.methods.eleet.model_input import ModelInput
from torch.utils.data import get_worker_info
import ast

from eleet.methods.eleet.value import MMValue


TMP_DIR_ROOT = Path.home() / ".eleet_tmpdir"
INFERENCE_BATCH_SIZE = 16


@define
class ELEETEngine(BaseEngine):
    model_name_or_path: str = field()
    model: ELEETInferenceModel = field(init=False)
    config: VerticalEleetConfig = field()
    tokenizer: PreTrainedTokenizerFast = field()
    num_preprocessing_workers_per_gpu: int = field(default=20)
    num_postprocessing_workers_per_gpu: int = field(default=6)
    name: str = field(default="ELEET")
    ctx: object = field(init=False, default=mp.get_context("spawn"))
    num_gpu = field(init=False, default=0)
    distribute_id_col_values_queue = field(init=False, default=None)
    model_output_queues = field(init=False, default=None)
    post_processing_done_queues = field(init=False, default=None)
    post_process_job_queue = field(init=False, default=None)
    post_processes = field(init=False, default=None)
    model_job_queue = field(init=False, default=None)
    model_input_queue = field(init=False, default=None)
    model_processes = field(init=False, default=None)
    dataset = field(init=False, default=None)
    data_loader = field(init=False, default=None)
    data_loader_iterator = field(init=False, default=None)
    model_input_job_queue = field(init=False, default=None)
    model_done_queue = field(init=False, default=None)
    tmp_dir = field(init=False, default=None)
    use_normed_conditions = False


    def __attrs_post_init__(self):
        self.model = ELEETInferenceModel.from_pretrained(model_name_or_path=self.model_name_or_path)
        self.model.eval()
        self.model.share_memory()

    def setup(self):
        print("Initiating setup")
        TMP_DIR_ROOT.mkdir(exist_ok=True)
        self.tmp_dir = TMP_DIR_ROOT / uuid.uuid4().hex
        (self.tmp_dir / "results").mkdir(exist_ok=True, parents=True)

        self.num_gpu = torch.cuda.device_count()

        if self.num_gpu == 1:
            self.num_preprocessing_workers_per_gpu = 1
            self.num_postprocessing_workers_per_gpu = 1

        total_num_post_processors = self.num_gpu * self.num_postprocessing_workers_per_gpu
        self.distribute_id_col_values_queue = DistributionQueue(
            num_workers=self.num_preprocessing_workers_per_gpu * self.num_gpu,
            multiprocessing_context=self.ctx, total_num_post_processors=total_num_post_processors, tmp_dir=self.tmp_dir
        )
        self.model_output_queues = [self.ctx.Queue(20 * torch.cuda.device_count()) for _ in range(self.num_gpu)]
        self.post_processing_done_queues = [self.ctx.Queue() for _ in range(self.num_gpu)]
        self.post_process_job_queue = self.ctx.Queue()
        self.post_processes = [self.ctx.Process(target=ELEETEngine.post_process, daemon=True, args=(
            out_queue, self.tokenizer, stop_queue, self.distribute_id_col_values_queue, self.post_process_job_queue,
            self.tmp_dir)
        ) for out_queue, stop_queue in zip(self.model_output_queues, self.post_processing_done_queues)
          for _ in range(self.num_postprocessing_workers_per_gpu)]
        for p in self.post_processes:
            print("Start post process")
            p.start()

        self.model_job_queue = self.ctx.Queue()
        self.model_input_queue = self.ctx.Queue(20 * torch.cuda.device_count())
        self.model_done_queue = self.ctx.Queue()
        self.model_processes = [self.ctx.Process(target=ELEETEngine.apply_model, daemon=True, args=(
            i, self.model, self.model_input_queue, out_queue, stop_queue, self.num_postprocessing_workers_per_gpu,
            self.model_job_queue, self.model_done_queue)
        ) for i, (out_queue, stop_queue) in enumerate(zip(self.model_output_queues, self.post_processing_done_queues))]
        for p in self.model_processes:
            print("Start model process")
            p.start()

        self.model_input_job_queue = self.ctx.Queue()
        self.dataset = ELEETInferenceDataset(self.model, self.model_input_job_queue, self.distribute_id_col_values_queue)
        self.data_loader = DataLoader(self.dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
                                      num_workers=self.num_preprocessing_workers_per_gpu * self.num_gpu,
                                      multiprocessing_context=self.ctx, persistent_workers=True)
        self.data_loader_iterator = iter(self.data_loader)
        print("Setup done")

    def execute(self, model_input: ModelInput, attributes, identifying_attribute, force_single_value_attributes,
                mode: EngineMode):
        os.environ["TOKENIZERS_PARALLELISM"] = "False"
        self.distribute_id_col_values_queue.run()

        for p in range(self.data_loader.num_workers):
            self.model_input_job_queue.put(model_input)
        for p in self.model_processes:
            self.model_job_queue.put(mode.value)
        for p in self.post_processes:
            self.post_process_job_queue.put((attributes, identifying_attribute, model_input.multi_iteration,
                                             model_input.report_table_name))

        for batch in self.data_loader_iterator:
            self.model_input_queue.put(batch)

        for p in self.model_processes:
            self.model_input_queue.put(None)

        num_model_processes_finished = 0
        while num_model_processes_finished < len(self.model_processes):
            self.model_done_queue.get()
            num_model_processes_finished += 1

        final_result = self.finalize(identifying_attribute, model_input.evidence_columns)
        self.data_loader_iterator = iter(self.data_loader)

        self.distribute_id_col_values_queue.join()
        return final_result

    def shutdown(self):
        print("Initiating shutdown")
        for p in self.post_processes:
            self.post_process_job_queue.put(None)
        for p in self.model_processes:
            self.model_job_queue.put(None)
        for p in self.post_processes:
            p.join()
        for p in self.model_processes:
            p.join()
        print("Shutdown done")

    def finalize(self, identifying_attribute, evidence_columns):
        results = list()
        for file in (self.tmp_dir / "results").iterdir():
            with file.open("rb") as f:
                result = pickle.load(f)
            file.unlink()
            results.append(result)
        index = pd.MultiIndex.from_frame(pd.concat(tuple(r.index.to_frame() for r in results))
                                         .sort_index().drop_duplicates())
        columns = sorted(set(c for r in results for c in r.columns if c != "__vec__"))
        final_result = pd.DataFrame(columns=columns, index=index).applymap(lambda x: MMValue())
        final_result["__vec__"] = 0
        for r in results:
            final_result.loc[r.index, "__vec__"] = 0
            final_result.loc[r.index, r.columns] += r

        final_result[columns].applymap(partial(MMValue.deduplicate, model=self.model))
        final_result.index.names = ("", identifying_attribute)
        if identifying_attribute is None or identifying_attribute in evidence_columns:
            final_result = final_result.droplevel(1).drop("__vec__", axis=1)
        else:
            final_result = final_result.drop(identifying_attribute, axis=1, errors="ignore") \
                .reset_index(identifying_attribute)
            final_result[identifying_attribute] = final_result[[identifying_attribute, "__vec__"]] \
                .apply(lambda x: MMValue([tuple(x.tolist())]), axis=1)
            final_result.drop("__vec__", inplace=True, axis=1)
            
        return final_result

    @staticmethod
    def apply_model(rank, model, model_input_queue, model_output_queue, post_processing_done_queue,
                    num_postprocessing_workers_per_gpu, model_job_queue, model_done_queue):
        while True:
            job = model_job_queue.get()
            if job is None:
                break
            mode = job
            model.set_union_mode(mode.union_mode)
            model.set_fast_mode(mode.fast_mode)
            model.set_index_mode(mode.index_mode)
            ELEETEngine._apply_model(rank=rank, model=model, model_input_queue=model_input_queue,
                                    model_output_queue=model_output_queue, post_processing_done_queue=post_processing_done_queue,
                                    num_postprocessing_workers_per_gpu=num_postprocessing_workers_per_gpu)
            print(f"Exit Model {rank}")
            model_done_queue.put(None)
        print(f"Shutdown model worker {rank}")

    @staticmethod
    def _apply_model(rank, model, model_input_queue, model_output_queue, post_processing_done_queue,
                     num_postprocessing_workers_per_gpu):
        device = f"cuda:{rank}"
        model = model.to(device)
        with torch.no_grad():
            while True:
                # print(f"{torch.cuda.memory_allocated(rank)}\t\t{in_queue.qsize()} - {out_queue.qsize()}")
                batch = model_input_queue.get()
                if batch is None:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                model_out = model(**batch)
                model_output_queue.put((model, batch, model_out))
        print(f"Done Model {rank}")
        for _ in range(num_postprocessing_workers_per_gpu):
            model_output_queue.put(None)
        model = batch = model_out = None
        num_consumers_finished = 0
        while num_consumers_finished < num_postprocessing_workers_per_gpu:
            post_processing_done_queue.get()
            num_consumers_finished += 1

    @staticmethod
    def post_process(model_output_queue, tokenizer, postprocessing_done_queue, distribute_id_col_values_queue,
                     post_process_job_queue, tmp_dir):
        while True:
            job = post_process_job_queue.get()
            if job is None:
                break
            columns, identifying_attribute, multi_iteration, report_table_name = job
            ELEETEngine._post_process(model_output_queue=model_output_queue, 
                                    tokenizer=tokenizer,
                                    distribute_id_col_values_queue=distribute_id_col_values_queue, 
                                    columns=columns,
                                    identifying_attribute=identifying_attribute,
                                    report_table_name=report_table_name,
                                    multi_iteration=multi_iteration, tmp_dir=tmp_dir)
            print("Exit Post Processing")
            postprocessing_done_queue.put(None)
        print("Shutdown post processing")

    @staticmethod
    def _post_process(model_output_queue, tokenizer, distribute_id_col_values_queue, columns,
                      identifying_attribute, report_table_name, multi_iteration, tmp_dir):
        result = pd.DataFrame(columns=columns)
        result.index = pd.MultiIndex.from_tuples([], names=("__row_id__", "__id_col__"))
        result_vec = pd.Series(name="__vec__", dtype=object)
        result_vec.index = pd.MultiIndex.from_tuples([], names=("__row_id__", "__id_col__"))
        result_vecs = list()
        new_row = pd.Series(data=[MMValue() for _ in range(len(columns))], index=columns)
        is_first_iteration = True
        while True:
            item = model_output_queue.get()
            if item is None:
                break
            model, batch, model_out = item
            masked_id_column_mask = (batch["id_col_value"][:, :, 0] == 103).any(1)
            if not multi_iteration:
                masked_id_column_mask[:] = False
            patch_iter_1 = patch_iter_2 = None
            if masked_id_column_mask.any():  # iteration 1 in complex alg
                patch_iter_1, _  = ELEETEngine.to_tuples(
                    batch, model_out, columns, model, tokenizer,
                    make_sure_cols_exists=[identifying_attribute], mask=masked_id_column_mask,
                    is_first_iteration=True, identifying_attribute=identifying_attribute,
                    report_table_name=report_table_name
                )
                is_first_iteration = False
            if not masked_id_column_mask.all():
                patch_iter_2, patch_vec = ELEETEngine.to_tuples(
                    batch, model_out, columns, model, tokenizer,
                    make_sure_cols_exists=[], mask=~masked_id_column_mask,
                    is_first_iteration=is_first_iteration,
                    identifying_attribute=identifying_attribute,
                    report_table_name=report_table_name
                )
            del model
            del batch
            del model_out
            del item
            del masked_id_column_mask

            if patch_iter_1 is not None:
                distribute_id_col_values_queue.put(patch_iter_1)
            if patch_iter_2 is not None:
                result_vecs.append(patch_vec)
                new_rows = sorted({(x, z) for x, _, z in patch_iter_2.index} - set(result.index))
                for r, n in new_rows:
                    result.loc[(r, n), :] = new_row

                patch_iter_2 = patch_iter_2.loc[patch_iter_2.index.levels[0], patch_iter_2.columns] \
                    .groupby(["__row_id__", "__id_col__"]).agg(partial(sum, start=MMValue()))


                result.loc[patch_iter_2.index, patch_iter_2.columns] += patch_iter_2

        distribute_id_col_values_queue.put(None)
        result_vec = pd.concat((result_vec, *result_vecs)) \
            .groupby(result_vec.index.names).agg(lambda x: pd.Series([x.iloc[0]] if len(x) else []))
        result = result.merge(result_vec, on=result_vec.index.names, how="left")
        with open(tmp_dir / "results" / uuid.uuid4().hex, "wb") as f:
            pickle.dump(result, f)

    @staticmethod
    def to_tuples(batch, model_out, columns, model, tokenizer, make_sure_cols_exists, mask, is_first_iteration,
                  identifying_attribute, report_table_name):
        with torch.no_grad():
            sd_pred, e_id, r_id, c_id = model.get_sd_pred(
                context_token_mask=batch["context_token_mask"][mask],
                context_encoding=model_out["context_encoding"][mask],
                schema_encoding=model_out["schema_encoding"][mask],
                table_encoding=model_out["table_encoding"][mask], 
                table_mask=batch["table_mask"][mask],
                query_mask=batch["query_mask"][mask],
                is_first_iteration=is_first_iteration
            )

            get_span_embeddings = partial(
                model.get_span_embeddings,
                context_encoding=model_out["deduplication_context_encoding"][mask],
                query_mask=batch["query_mask"][mask],
            )

            value_iterator = ELEETEngine.get_value_iterator(
                get_span_embeddings=get_span_embeddings,
                tokenizer=tokenizer,
                sd_pred=sd_pred,
                e_id=e_id,
                r_id=r_id,
                c_id=c_id,
                window_offset=batch["window_offset"][mask],
                row_id=batch["row_id"][mask],
                id_col_value=batch["id_col_value"][mask],
                id_col_vec=batch["id_col_vec"][mask]
            )

            patch, id_col_vecs = ELEETEngine.get_patch(
                tokenizer=tokenizer,
                columns=columns,
                input_ids=batch["input_ids"][mask],
                col_token_pos_to_col_ids=batch["column_token_position_to_column_ids"][mask],
                value_iterator=value_iterator,
                identifying_attribute=identifying_attribute,
                report_table_name=report_table_name)

            patch, patch_vec = ELEETEngine.finalize_patch(
                make_sure_cols_exists=make_sure_cols_exists,
                patch=patch,
                window_offset=batch["window_offset"][mask],
                row_id=batch["row_id"][mask],
                id_col_vecs=id_col_vecs)
        return patch, patch_vec

    @staticmethod
    def get_value_iterator(get_span_embeddings, tokenizer, sd_pred, e_id, r_id, c_id, window_offset, row_id,
                           id_col_value, id_col_vec):
        x_id, t_start_id = torch.where(sd_pred == 2)
        t_end_id = ELEETEngine.get_token_end_ids(sd_pred, x_id, t_start_id)
        span_embeddings = get_span_embeddings(token_start_id=t_start_id,
                                              token_end_id=t_end_id,
                                              sample_id=e_id[x_id],
                                              row_id=r_id[x_id])

        window_offsets = window_offset[e_id, r_id][x_id].tolist()
        row_nums = row_id[e_id, r_id][x_id].tolist()
        id_col_vals = [tokenizer.decode(v, skip_special_tokens=True)
                                        for v in id_col_value[e_id, r_id][x_id]]
        id_col_vecs = id_col_vec[e_id, r_id][x_id]
        value_tuple = namedtuple("value_tuple", ["sample_id", "row_id", "col_id", "token_start", "token_end",
                                                 "window_offset", "row_number", "id_col_value", "id_col_vec",
                                                 "span_embedding"])
        value_iterator = map(lambda x: value_tuple(*x), zip(
            e_id[x_id], r_id[x_id], c_id[x_id],
            t_start_id, t_end_id, window_offsets, row_nums, id_col_vals, id_col_vecs, span_embeddings
        ))
            
        return value_iterator

    @staticmethod
    def finalize_patch(make_sure_cols_exists, patch, window_offset, row_id, id_col_vecs):
        for col in make_sure_cols_exists:
            for row_num, offset in zip(row_id.flatten().tolist(), window_offset.flatten().tolist()):
                if row_num >= 0 and (row_num, offset, '') not in patch[col]:
                    patch[col][(row_num, offset, '')] = []

        patch = pd.DataFrame(patch).fillna(0).applymap(lambda x: MMValue() if x == 0 else x)
        patch_vec = pd.Series(id_col_vecs, name="__vec__", dtype=object)

        if len(patch) == 0:
            patch.index = pd.MultiIndex.from_tuples([], names=("__row_id__", "__offset__", "__id_col__"))
        if len(patch_vec) == 0:
            patch_vec.index = pd.MultiIndex.from_tuples([], names=("__row_id__", "__id_col__"))
        patch.index.names = ("__row_id__", "__offset__", "__id_col__")
        patch_vec.index.names = ("__row_id__", "__id_col__")
        return patch, patch_vec

    @staticmethod
    def get_patch(tokenizer, columns, input_ids, col_token_pos_to_col_ids, value_iterator,
                  identifying_attribute, report_table_name):
        result_patch = dict()
        id_col_vecs = dict()
        col_dict = {c.lower().replace(" ", ""): c for c in columns}
        if identifying_attribute is not None:
            id_col = f"{report_table_name[1]} {identifying_attribute}".lower().replace(" ", "")
            col_dict[id_col] = identifying_attribute
        for v in value_iterator:  # build result data frame
            offset, row_num = v.window_offset[0], v.row_number[0]
            val = "##"
            ts = v.token_start
            while val.startswith("##") and ts >= 0:
                val = tokenizer.decode(input_ids[v.sample_id, v.row_id, ts:v.token_end].to("cpu"))
                ts -= 1
            col = tokenizer.decode(
                input_ids[v.sample_id, v.row_id][col_token_pos_to_col_ids[v.sample_id, v.row_id] == v.col_id].to("cpu")
            ).split("|")[0].strip()
            col = col_dict[col.replace(" ", "")]
            result_patch[col] = result_patch.get(col, dict())
            result_patch[col][(row_num, offset, v.id_col_value)] = result_patch[col].get(
                (row_num, offset, v.id_col_value), MMValue())
            result_patch[col][(row_num, offset, v.id_col_value)].append((val, v.span_embedding.to("cpu")))
            id_col_vecs[(row_num, v.id_col_value)] = v.id_col_vec.to("cpu")
        return result_patch, id_col_vecs

    @staticmethod
    def get_token_end_ids(sd_pred, x_id, t_start_id):
        current = t_start_id.clone() + 1
        t_end_id = torch.zeros_like(t_start_id, dtype=int)
        done = torch.zeros_like(t_start_id, dtype=bool)

        while (~done).any():
            values = sd_pred[x_id, current.clip(0, sd_pred.shape[1] - 1)]
            values[current >= sd_pred.shape[1]] = 0
            this_done = values != 1
            new_done = (~done) & this_done
            t_end_id[new_done] = current[new_done] 
            done = done | this_done
            current += 1
        return t_end_id

    def aggregate(self, col_values):
        values, assignment = MMValue(col_values.tolist()).deduplicate(self.model, return_assignment=True,
                                                                      linkage="complete")
        groups = dict()
        for v in np.unique(assignment):
            ids = tuple(np.where(assignment == v)[0])
            groups[ids] = values[v][0]
        return groups

    def build_index(self, extract_attributes, operands, model_input):
        start_time = datetime.now()
        result = self.execute(model_input, extract_attributes, operands[0].identifying_attribute,
                                operands[0].force_single_value_attributes, mode=EngineMode.INDEX)
        values = result[extract_attributes[-1]].explode()
        values = values[~values.isna()]
        _, assignment = MMValue(values.tolist()).deduplicate(self.model, return_assignment=True,
                                                             linkage="complete")
        result_index = defaultdict(set)
        result_synonyms = dict()
        for v in sorted(set(assignment)):
            mask = assignment == v
            synonyms = set(values[mask].apply(lambda x: x[0]))
            for synonym in synonyms:
                result_index[synonym] |= set(values.iloc[mask].index)
                result_synonyms[synonym] = synonyms
        end_time = datetime.now()
        self.index_build_time = end_time - start_time
        return result_index, result_synonyms


@define
class DistributionQueue():
    num_workers = field()
    multiprocessing_context = field()
    total_num_post_processors = field()
    tmp_dir = field()
    in_queue = field(init=False, default=None)
    collect_row_id_queues = field(init=False, default=Factory(list))
    distribution_queues = field(init=False, default=Factory(list))
    row_id_map = field(init=False, default=Factory(dict))
    thread_manage_collect_row_id_queues = field(init=False, default=None)
    thread_manage_distribution_queues = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.collect_row_id_queues = [self.multiprocessing_context.Queue(100) for _ in range(self.num_workers)]
        self.distribution_queues = [self.multiprocessing_context.Queue() for _ in range(self.num_workers)]
        self.in_queue = self.multiprocessing_context.Queue(100)

    def put(self, item):
        if item is None:
            self.in_queue.put((None, None))
            return

        filename = self.tmp_dir / uuid.uuid4().hex
        with open(filename, "wb") as f:
            pickle.dump(item, f)
        self.in_queue.put((filename, item.index))

    def put_row_ids(self, row_ids):
        worker_info = get_worker_info()
        self.collect_row_id_queues[worker_info.id].put(row_ids)

    def get(self):
        worker_info = get_worker_info()
        filename =  self.distribution_queues[worker_info.id].get()
        with open(filename, "rb") as f:
            item = pickle.load(f)
        os.unlink(filename)
        return item

    def run(self):
        self.thread_manage_collect_row_id_queues = Thread(target=self._manage_collect_row_id_queues, daemon=True)
        self.thread_manage_distribution_queues = Thread(target=self._manage_distribution_queues, daemon=True)
        self.thread_manage_collect_row_id_queues.start()
        self.thread_manage_distribution_queues.start()
    
    def join(self):
        self.thread_manage_collect_row_id_queues.join()
        self.thread_manage_distribution_queues.join()

    def _manage_collect_row_id_queues(self):
        num_workers_first_iter_finished = 0
        while True:
            for worker_id, q in enumerate(self.collect_row_id_queues):
                try:
                    row_ids = q.get_nowait()
                except queue.Empty:
                    sleep(0.1)
                    continue

                if row_ids is None:
                    num_workers_first_iter_finished += 1
                    if num_workers_first_iter_finished == self.num_workers:
                        print("Exit Distribution Queue Row ID collector.")
                        return
                    continue
    
                for r_id in row_ids:
                    self.row_id_map[r_id] = worker_id

    def _manage_distribution_queues(self):
        num_post_processors_finished = 0
        while True:
            item, item_index = self.in_queue.get()

            if item is None:
                num_post_processors_finished += 1
                if num_post_processors_finished == self.total_num_post_processors:
                    print("Exit Distribution Queue Item distributor.")
                    return
                continue

            try:
                worker_id = next(self.row_id_map[r_id] for r_id, _, _ in item_index
                                 if r_id >= 0 and r_id in self.row_id_map)
                self.distribution_queues[worker_id].put(item)
            except StopIteration:
                self.in_queue.put((item, item_index))

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__ if "thread" not in k and "__" not in k}

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        for k, v in state.items():
            setattr(self, k, v)
