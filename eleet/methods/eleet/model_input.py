from copy import deepcopy
import os
from attr import Factory
from attrs import define, field
from torch.utils.data import get_worker_info
import torch
import tqdm

from eleet.methods.eleet.value import MMValue


@define
class ModelInput():
    iterator = field()
    identifying_attribute = field()
    report_table_name = field()
    evidence_columns = field()
    tokenizer = field()
    config = field()
    index_mode = field(default=False)
    fill_values = field(init=False, default=None)
    worker_info = field(init=False, default=None)
    tensor_dict_cache = field(init=False, default=Factory(dict))
    value_cache = field(init=False, default=Factory(dict))
    distribution_queue = field(init=False, default=Factory(dict))
    model = field(init=False, default=None)

    @property
    def multi_iteration(self):
        return self.identifying_attribute not in self.evidence_columns \
            and self.identifying_attribute is not None and not self.index_mode

    def __iter__(self):
        from eleet.methods.eleet.engine import INFERENCE_BATCH_SIZE
        from eleet.methods.eleet.preprocessor import FILL_VALUES
        self.fill_values = FILL_VALUES
        assert self.distribution_queue is not None
        self.worker_info = get_worker_info()

        num_first_iter = 0
        for x in self._first_iter():
            yield x
            num_first_iter += 1

        if self.multi_iteration:
            while num_first_iter < INFERENCE_BATCH_SIZE:
                yield {'input_ids': torch.zeros([3, 512], dtype=int),
                       'token_type_ids': torch.zeros([3, 512], dtype=int),
                       'sequence_mask': torch.zeros([3, 512], dtype=int),
                       'column_token_position_to_column_ids': torch.full([3, 512], -1, dtype=int),
                       'context_token_positions': torch.zeros([3, 512], dtype=int),
                       'context_token_mask': torch.zeros([3, 512], dtype=int),
                       'table_mask': torch.zeros([3, 11], dtype=int),
                       'query_mask': torch.full([3, 11], -1, dtype=int),
                       'row_id': torch.full([3, 1], -1, dtype=int),
                       'col_offset': torch.zeros([3, 1], dtype=int),
                       'id_col_value': torch.zeros([3, 20], dtype=int),
                       'id_col_vec': torch.zeros([3, self.config.hidden_size], dtype=float),
                       'window_offset': torch.zeros([3, 1], dtype=int)}
                num_first_iter += 1

            yield from self._subsequent_iter()

    def set_distribute_id_col_values_queue(self, distribution_queue):
        self.distribution_queue = distribution_queue

    def set_model(self, model):
        self.model = model

    def _first_iter(self):
        for tensor_dict in self.iterator(worker_id=self.worker_info.id,
                                         num_workers=self.worker_info.num_workers):
            if self.multi_iteration:
                self.cache_tensor_dict(tensor_dict)
                yield from self._adjust_first_iter(tensor_dict)
            else:
                yield tensor_dict
        self.distribution_queue.put_row_ids(None)

    def cache_tensor_dict(self, tensor_dict):
        row_ids, col_offsets, window_offsets = zip(
            *tuple((x, y, z) for x, y, z in zip(tensor_dict["row_id"].flatten().tolist(),
                                                tensor_dict["col_offset"].flatten().tolist(),
                                                tensor_dict["window_offset"].flatten().tolist()) if x >= 0)
        )
        self.distribution_queue.put_row_ids(row_ids)
        for r_id, col_offset, window_offset in zip(row_ids, col_offsets, window_offsets):
            mask = (tensor_dict["row_id"] == r_id).flatten()
            example_rows_mask = (tensor_dict["row_id"] == -1).flatten()
            self.tensor_dict_cache[r_id] = self.tensor_dict_cache.get(r_id, dict())
            self.tensor_dict_cache[r_id][col_offset] = self.tensor_dict_cache[r_id].get(col_offset, dict())
            self.tensor_dict_cache[r_id][col_offset][window_offset] = (
                {k: v[mask] for k, v in tensor_dict.items()},
                {k: v[example_rows_mask] for k, v in tensor_dict.items()}
            )

    def _subsequent_iter(self):
        with tqdm.tqdm(total=len(self.tensor_dict_cache), desc=f"Iteration 2, Worker {self.worker_info.id}",
                       position=self.worker_info.id) as pbar:
            while len(self.tensor_dict_cache):
                result_first_iter = self.distribution_queue.get()[self.identifying_attribute]
                for (row_id, window_offset, _), values in list(result_first_iter.iteritems()):
                    self.value_cache[row_id] = self.value_cache.get(row_id, dict())
                    self.value_cache[row_id][window_offset] = values

                    if len(self.value_cache[row_id]) == len(self.tensor_dict_cache[row_id][0]):
                        values = sum(self.value_cache[row_id].values(), start=MMValue())
                        values.deduplicate(self.model)
                        yield from self._adjust_second_iter(values, row_id)
                        del self.tensor_dict_cache[row_id]
                        pbar.update(1)
        print(f"Exit Worker {self.worker_info.id}")

    def _adjust_first_iter(self, tensor_dict):
        if tensor_dict["col_offset"][0, 0] == 0:
            yield tensor_dict
    
    def _adjust_second_iter(self, values, row_id):
        for col_offset in self.tensor_dict_cache[row_id]:
            for window_offset in self.tensor_dict_cache[row_id][col_offset]:
                tensor, example_rows = self.tensor_dict_cache[row_id][col_offset][window_offset]
                value_mask = tensor["column_token_position_to_column_ids"] == 0
                token_id = torch.where(value_mask)[1][-1]
                working_set = [example_rows] if example_rows["input_ids"].shape[0] > 0 else []
                assert tensor["input_ids"][0, token_id] == 103
                for value, vec in values:
                    new_tensor = deepcopy(tensor)
                    value_tokenized = self.tokenizer(value, add_special_tokens=False)["input_ids"]
                    new_tensor["id_col_value"][0, :len(value_tokenized)] = torch.tensor(value_tokenized, dtype=int)
                    new_tensor["id_col_vec"][0] = vec
                    self.do_value_inserts(old_tensor=tensor, token_id=token_id, value=value, new_tensor=new_tensor)
                    self.fix_not_ending_with_sep(new_tensor)
                    self.fix_table_and_query_mask(new_tensor)
                    working_set.append(new_tensor)
                    yield from self.combine_to_table(working_set, example_rows)
                yield from self.combine_to_table(working_set, example_rows, force=True)

    def combine_to_table(self, working_set, example_rows, force=False):
        num_rows = sum([x["input_ids"].shape[0] for x in working_set])
        force = force and num_rows > example_rows["input_ids"].shape[0]
        while num_rows >= self.config.sample_row_num or force:
            merged = {k: torch.concat(tuple(x[k] for x in working_set)) for k in working_set[0].keys()}
            tensor = {k: v[:self.config.sample_row_num] for k, v in merged.items()}
            remaining = {k: v[self.config.sample_row_num:] for k, v in merged.items()}

            pad_amount = self.config.num_vertical_layers - len(tensor["input_ids"])
            tensor = {k: torch.nn.functional.pad(v, (0, 0, 0, pad_amount),
                                                 value=self.fill_values.get(k, 0)) for k, v in tensor.items()}

            yield tensor
            working_set.clear()
            if example_rows["input_ids"].shape[0] > 0:
                working_set.append(example_rows)
            if remaining["input_ids"].shape[0] > 0:
                working_set.append(remaining)
            force = False
            num_rows = sum([x["input_ids"].shape[0] for x in working_set])


    def do_value_inserts(self, old_tensor, token_id, value, new_tensor):
        tokens = self.tokenizer(value, add_special_tokens=False,
                                        return_tensors="pt")["input_ids"][:self.config.max_cell_len][0]
        for k, t in new_tensor.items():
            if t.shape[1] > 100 and t.shape[1] != self.config.hidden_size:
                t[0, token_id + len(tokens):] = \
                    t[0, token_id + 1: old_tensor["input_ids"].size(1) - len(tokens) + 1].clone()
                if k == "input_ids":
                    t[0, token_id : token_id + len(tokens)] = tokens
                else:
                    t[0, token_id: token_id + len(tokens)] = t[0, token_id].clone()

    def fix_table_and_query_mask(self, new_tensor):
        num_inactive_cols = (
            new_tensor["table_mask"].size(-1) - new_tensor["column_token_position_to_column_ids"].max() - 1
        )
        if num_inactive_cols:
            new_tensor["table_mask"][0, -num_inactive_cols:] = 0

        new_tensor["query_mask"][(new_tensor["query_mask"] == 0) | (new_tensor["table_mask"] == 0)] = -1

    def fix_not_ending_with_sep(self, new_tensor):
        fill_values = dict(
            column_token_position_to_column_ids=-1,
        )
        if new_tensor["column_token_position_to_column_ids"][0, -1] != -1:
            rm_col = new_tensor["column_token_position_to_column_ids"][0, -1]
            rm_mask = rm_col == new_tensor["column_token_position_to_column_ids"][0]

            for k, t in new_tensor.items():
                if t[0].shape == rm_mask.shape:
                    t[0, rm_mask] = fill_values.get(k, 0)

@define
class FinetuningInput():
    iterator = field()
    identifying_attribute = field()
    evidence_columns = field()
    tokenizer = field()
    config = field()
    max_split_size = field()
    worker_info = field(init=False, default=None)

    def __iter__(self):
        worker_id, num_workers = self.get_worker_info()
        for tensor_dict in self.iterator(worker_id=worker_id,
                                         num_workers=num_workers):
            yield tensor_dict

    def get_worker_info(self):
        self.worker_info = get_worker_info()
        worker_id = self.worker_info.id
        num_workers = self.worker_info.num_workers
        return worker_id, num_workers

    def iter_with_specified_split_size(self, split_size):
        worker_id, num_workers = self.get_worker_info()
        for tensor_dict in self.iterator(worker_id=worker_id,
                                         num_workers=num_workers,
                                         limit=split_size):
            yield tensor_dict
