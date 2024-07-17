from contextlib import contextmanager
from functools import partial, reduce
from math import ceil

import numpy as np
import pandas as pd
import torch
import tqdm
from eleet_pretrain.model.config import VerticalEleetConfig
from eleet.methods.base_preprocessor import BasePreprocessor
from attrs import define, field
from eleet.methods.eleet.input_formatter import ELEETInputFormatter
from torch import nn
from transformers import PreTrainedTokenizerFast

from eleet.methods.eleet.model_input import FinetuningInput, ModelInput


FILL_VALUES = {
    "column_token_position_to_column_ids": -1,
    "query_mask": -1,
    "row_id": -1
}


@define
class ELEETPreprocessor(BasePreprocessor):
    tokenizer: PreTrainedTokenizerFast = field()
    config: VerticalEleetConfig = field()
    input_formatter: ELEETInputFormatter = field(init=False)
    max_columns: int = field(default=11)
    max_query_columns: int = field(default=7)
    rng = field(init=False, default=np.random.default_rng(42))
    finetuning_independent_of_operator = False

    def __attrs_post_init__(self):
        self.config.max_num_cols = self.max_columns
        self.config.max_query_cols = self.max_query_columns
        self.input_formatter = ELEETInputFormatter(self.config, self.tokenizer)

    def slice_table(self, num_rows, num_rows_per_sample, worker_id, num_workers, shuffle, iterate_indefinitely):
        num = {num_rows_per_sample: num_rows // num_rows_per_sample}
        if num_rows % num_rows_per_sample:
            num[num_rows % num_rows_per_sample] = 1
        if 1 in num and num[num_rows_per_sample] > 0 and num_rows_per_sample >= 3:
            num[1] = 0
            num[2] = 1
            num[num_rows_per_sample] -= 1
            num[num_rows_per_sample - 1] = num.get(num_rows_per_sample - 1, 0) + 1

        slices = list(self._slice_table(num, worker_id, num_workers))
        for epoch in range((2 ** 32) if iterate_indefinitely else 1):
            order = np.arange(len(list(slices)))
            if shuffle:
                self.rng.shuffle(order)
            if num_workers < 10 or int(worker_id) == 0:
                order = tqdm.tqdm(order, position=worker_id, desc=f"Epoch {epoch}, Iteration 1, Worker {worker_id}")
            for i in order:
                yield slices[i]


    def _slice_table(self, num, worker_id, num_workers):
        i = 0
        for k, v in sorted(num.items(), reverse=True):
            for j in range(worker_id * k, v * k, num_workers * k):
                yield np.arange(i + j, i + j + k)
            i += v * k

    def slice_columns(self, num_query_cols, identify_attr_offset, num_attrs_per_pass, shuffle):
        num_cols = num_query_cols - identify_attr_offset
        if not shuffle:
            for i in range(0, num_cols, num_attrs_per_pass):
                col_slice = range(i + identify_attr_offset,
                                  min(i + num_attrs_per_pass + identify_attr_offset, num_query_cols))
                yield col_slice
        else:
            order = np.arange(num_cols) + identify_attr_offset
            self.rng.shuffle(order)
            num_selected_cols = self.rng.integers(2, self.max_query_columns)
            yield order[:num_selected_cols]

    def iter_data(self, data, example_rows, extract_attributes, multi_iteration,
                  worker_id, num_workers, limit=2**32, shuffle=False, iterate_indefinitely=False):
        multi_iteration = int(multi_iteration)
        example_rows = example_rows.data if example_rows else pd.DataFrame(columns=extract_attributes)
        num_example_tuples = min(2, len(example_rows))
        data_tuples_per_sample = 3 - num_example_tuples
        example_tuple_slice = self.get_example_slice(example_rows, num_example_tuples, shuffle=shuffle, limit=limit)
        example_tuples = example_rows.iloc[example_tuple_slice]
        num_query_cols = len(extract_attributes)
        num_passes = ceil((num_query_cols - multi_iteration) /
                          (self.max_query_columns - multi_iteration))
        num_attrs_per_pass = ceil((num_query_cols - multi_iteration) / num_passes)
        limit = len(data.data.loc[data.data.index.unique(0)[:limit]])
        for row_slice in self.slice_table(min(len(data), limit), data_tuples_per_sample, worker_id, num_workers,
                                          shuffle=shuffle, iterate_indefinitely=iterate_indefinitely):
            for i, col_slice in enumerate(self.slice_columns(num_query_cols, multi_iteration, num_attrs_per_pass,
                                                             shuffle=shuffle)):
                result_rows = data.data.iloc[row_slice]
                result_cols = extract_attributes[:multi_iteration] + [extract_attributes[c] for c in col_slice]
                table_name = f"{','.join(map(str, row_slice))}-{i}"
                row_slice_full = np.hstack((np.array([-1] * num_example_tuples), row_slice))
                yield table_name, result_rows, result_cols, example_tuples, row_slice_full

    def get_example_slice(self, example_rows, num_example_tuples, shuffle, limit):
        if not shuffle:
            example_tuple_slice = example_rows.applymap(bool).sum(1).reset_index(drop=True) \
                .sort_values(ascending=False)[:num_example_tuples].index
            return example_tuple_slice
        else:
            limit = len(example_rows.loc[example_rows.index.unique(0)[:limit]])
            return self.rng.choice(min(len(example_rows), limit), size=num_example_tuples, replace=False)

    @contextmanager
    def compute_model_input(self, data, report_column, report_table_name, extract_attributes, identifying_attribute,
                            example_rows, multi_table, limit=2**32, index_mode: bool=False):
        index_levels = data.data.index.levels if hasattr(data.data.index, "levels") else [data.data.index]
        evidence_columns = [l.name for l in index_levels if l.dtype != int] \
            + [c for c in data.data.columns if c != report_column]

        yield ModelInput(
            iterator=partial(self._compute_model_input,
                             data=data,
                             report_column=report_column,
                             extract_attributes=extract_attributes,
                             evidence_columns=evidence_columns,
                             identifying_attribute=identifying_attribute,
                             example_rows=example_rows,
                             alignments=None,
                             normed=None,
                             limit=limit,
                             shuffle=False,
                             report_table_name=report_table_name,
                             multi_table=multi_table,
                             index_mode=index_mode),
            identifying_attribute=identifying_attribute,
            report_table_name=report_table_name,
            evidence_columns=evidence_columns,
            tokenizer=self.tokenizer,
            config=self.config,
            index_mode=index_mode
        )

    @contextmanager
    def compute_finetuning_data(self, data, report_column, report_table_name, extract_attributes, identifying_attribute,
                                example_rows, alignments, normed, multi_table, shuffle):
        index_levels = data.data.index.levels if hasattr(data.data.index, "levels") else [data.data.index]
        evidence_columns = [l.name for l in index_levels if l.dtype != int] \
            + [c for c in data.data.columns if c != report_column]

        yield FinetuningInput(
            iterator=partial(self._compute_model_input,
                             data=data,
                             report_column=report_column,
                             extract_attributes=extract_attributes,
                             evidence_columns=evidence_columns,
                             identifying_attribute=identifying_attribute,
                             example_rows=example_rows,
                             alignments=alignments,
                             normed=normed,
                             shuffle=shuffle,
                             iterate_indefinitely=shuffle,
                             report_table_name=report_table_name,
                             multi_table=multi_table),
            identifying_attribute=identifying_attribute,
            evidence_columns=evidence_columns,
            tokenizer=self.tokenizer,
            config=self.config,
            max_split_size=len(data.data.index.unique())
        )

    def _compute_model_input(self, data, report_column, report_table_name, extract_attributes, evidence_columns,
                             identifying_attribute, example_rows, worker_id, num_workers, alignments, normed,
                             shuffle, multi_table, limit=None, iterate_indefinitely=False, index_mode=False):
        identifying_attribute_in_evidence = identifying_attribute in evidence_columns
        multi_iteration = not identifying_attribute_in_evidence and identifying_attribute is not None and not index_mode
        identifying_attribute_in_table = f"{report_table_name[1]} {identifying_attribute}" \
            if multi_table and identifying_attribute is not None else identifying_attribute
        iterator = self.iter_data(data=data,
                                  example_rows=example_rows,
                                  extract_attributes=extract_attributes,
                                  multi_iteration=multi_iteration,
                                  worker_id=worker_id,
                                  num_workers=num_workers,
                                  limit=limit,
                                  shuffle=shuffle,
                                  iterate_indefinitely=iterate_indefinitely)
        for idx, row_slice, query_col_slice, example_slice, row_ids in iterator:
            num_evidence_columns = max(0, min(len(evidence_columns), self.max_columns - len(query_col_slice)))
            queries = self._generate_queries(row_slice, query_col_slice, len(example_slice),
                                             identifying_attribute_in_evidence)
            table, texts = self._generate_tables_and_texts(
                row_slice=row_slice,
                identifying_attribute=identifying_attribute,
                identifying_attribute_in_evidence=identifying_attribute_in_evidence,
                report_column=report_column,
                evidence_columns=evidence_columns[:num_evidence_columns],
                queried_cols=query_col_slice,
                example_slice=example_slice,
                identifying_attribute_in_table=identifying_attribute_in_table)
            answers = header_queries = None

            if alignments is not None and normed is not None:
                queries, header_queries, answers = self.get_finetuning_input(
                    identifying_attribute=identifying_attribute,
                    alignments=alignments,
                    normed=normed,
                    multi_iteration=multi_iteration,
                    row_slice=row_slice,
                    queries=queries,
                    table=table,
                    num_example_rows=len(example_slice),
                    identifying_attribute_in_table=identifying_attribute_in_table,
                    extract_attributes=extract_attributes)

            instances = self.input_formatter.get_instances(queries=queries, table=table, texts=texts,
                                                           table_name=idx, row_ids=row_ids, answers=answers,
                                                           header_queries=header_queries)

            yield from self.to_tensor_dict(instances)

    def get_finetuning_input(self, identifying_attribute, alignments, normed,
                             multi_iteration, row_slice, queries, table, num_example_rows,
                             identifying_attribute_in_table, extract_attributes):
        iteration = identifying_attribute_value = -1
        row_slice_index = [i for i in row_slice.index if i in alignments.data.index]
        alignments_slice = alignments.data.loc[row_slice_index, extract_attributes]
        normed_slice = normed.data.loc[row_slice_index, extract_attributes]

        if multi_iteration:
            iteration = self.rng.integers(2) if len(normed_slice) else 0
            if iteration > 0:
                normed_slice = normed_slice[normed_slice[identifying_attribute].apply(len).apply(bool)]
                identifying_attribute_value = self.rng.integers(len(normed_slice))
                identifying_attribute_value = normed_slice.iloc[identifying_attribute_value] \
                            [identifying_attribute][0]
                
        all_queries = queries
        try:
            agg_alignments_slice = alignments.data.loc[row_slice.index.get_level_values(0), extract_attributes] \
                .groupby(row_slice.index.names[0]).agg(partial(reduce, lambda a, b: a + b))
            agg_normed_slice = normed.data.loc[row_slice.index.get_level_values(0), extract_attributes] \
                .groupby(row_slice.index.names[0]).agg(partial(reduce, lambda a, b: a + b))
        except KeyError:
            assert len(row_slice.index.get_level_values(0)) == 1
            agg_alignments_slice = alignments.data.iloc[[0]][extract_attributes] \
                .groupby(row_slice.index.names[0]).agg(partial(reduce, lambda a, b: a + b)).applymap(lambda x: [])
            agg_normed_slice = normed.data.iloc[[0]][extract_attributes] \
                .groupby(row_slice.index.names[0]).agg(partial(reduce, lambda a, b: a + b)).applymap(lambda x: [])

        if iteration == 1:
            table.iloc[-1][identifying_attribute_in_table] = identifying_attribute_value
            queries = queries[queries["col_id"] != 0]
            alignments_slice = alignments_slice.loc[(slice(None), identifying_attribute_value), :]
            normed_slice = normed_slice.loc[(slice(None), identifying_attribute_value), :]
        elif iteration == 0:
            queries = queries[queries["col_id"] == 0]
            alignments_slice = agg_alignments_slice
            normed_slice = agg_normed_slice
        elif iteration == -1:
            agg_alignments_slice = agg_alignments_slice.loc[alignments_slice.index.get_level_values(0)]
            agg_normed_slice = agg_normed_slice.loc[normed_slice.index.get_level_values(0)]

        if identifying_attribute is not None:
            for x in (alignments_slice, normed_slice, agg_alignments_slice, agg_normed_slice):
                x.rename({identifying_attribute: identifying_attribute_in_table}, axis=1, inplace=True)

        answers = self._generate_answers(
            queries=queries,
            table=table,
            alignments_slice=alignments_slice,
            normed_slice=normed_slice,
            num_example_rows=num_example_rows,
        )
        header_queries = self._generate_answers(
            queries=all_queries,
            table=table,
            alignments_slice=agg_alignments_slice,
            normed_slice=agg_normed_slice,
            num_example_rows=num_example_rows
        )
        return queries, header_queries, answers

    def _generate_queries(self, row_slice, query_col_slice, num_example_rows, identifying_attribute_in_evidence):
        queries = pd.DataFrame(dict(
            row_id=num_example_rows + np.repeat(np.arange(len(row_slice)), len(query_col_slice)),
            col_id=identifying_attribute_in_evidence + np.tile(np.arange(len(query_col_slice)), len(row_slice)),
            query_id=np.arange(len(row_slice) * len(query_col_slice))
        ))
        return queries

    def _generate_tables_and_texts(self, row_slice, identifying_attribute, identifying_attribute_in_evidence,
                                   report_column, evidence_columns, queried_cols, example_slice,
                                   identifying_attribute_in_table):
        row_slice = row_slice.reset_index()
        example_slice = example_slice.reset_index()
        for c in queried_cols:
            row_slice[c] = "[MASK]"
        attributes = ([identifying_attribute] if identifying_attribute_in_evidence else []) + queried_cols + \
            [e for e in evidence_columns if e != identifying_attribute]
        if len(example_slice):
            table = pd.concat((example_slice[attributes], row_slice[attributes])).reset_index(drop=True)
            texts = pd.concat((pd.Series([""] * len(example_slice), name=report_column),
                            row_slice[report_column])).reset_index(drop=True)
        else:
            table = row_slice[attributes]
            texts = row_slice[report_column]
        
        if identifying_attribute is not None:
            table = table.rename({identifying_attribute: identifying_attribute_in_table}, axis=1)
        return table, texts

    def _generate_answers(self, queries, table, alignments_slice, normed_slice, num_example_rows):
        answers = pd.concat(
            (queries,
             queries.apply(
                 lambda x: np.arange(len(alignments_slice[table.columns[x["col_id"]]].iloc[x["row_id"]
                                                                                           - num_example_rows])),
                 axis=1)),
        axis=1).explode(0)
        answers = answers[answers[0].notna()]
        answers = answers.rename({0: "answer_id"}, axis=1)
        if len(answers):
            answers = pd.concat(
                (answers,
                answers.apply(
                    lambda x: alignments_slice[table.columns[x["col_id"]]].iloc[x["row_id"]
                                                                                - num_example_rows][x["answer_id"]],
                    axis=1),
                answers.apply(
                    lambda x: normed_slice[table.columns[x["col_id"]]].iloc[x["row_id"]
                                                                            - num_example_rows][x["answer_id"]],
                    axis=1)),
            axis=1)
        else:
            answers[0] = []
            answers[1] = []
        answers["start"] = answers[0].apply(lambda x: x[0])
        answers["end"] = answers[0].apply(lambda x: x[1])
        answers.drop(0, axis=1, inplace=True)
        answers.drop("answer_id", axis=1, inplace=True)
        answers.rename({1: "normed"}, axis=1, inplace=True)
        answers.drop_duplicates(inplace=True)
        return answers

    def to_tensor_dict(self, instances):
        max_lens = {
            "table_mask": (self.max_columns,),
            "query_mask": (self.max_columns,),
            "col_offset": (1,),
            "row_id": (1,),
            "id_col_value": (20,),
            "id_col_vec": (self.config.hidden_size,),
            "deduplication_labels": (self.config.max_num_deduplication_labels_per_row,
                                     2 + self.config.max_cell_len + self.config.deduplication_max_normed_len),
            "query_labels": (self.max_columns, self.input_formatter.config.max_sequence_len),
            "header_query_labels": (self.max_columns, self.input_formatter.config.max_sequence_len),
        }
        def pad(tensors, key):
            rows_missing = self.input_formatter.config.sample_row_num - len(tensors)
            empty = tuple(torch.tensor([], dtype=tensors[0].dtype).reshape(*tuple(0 for _ in range(tensors[0].dim())))
                          for _ in range(rows_missing))
            tensors = tensors + empty
            max_len = max_lens.get(key, (self.input_formatter.config.max_sequence_len,))
            fill_value = FILL_VALUES.get(key, 0)
            return tuple(nn.functional.pad(tensor, get_pad(max_len, tensor), value=fill_value) for tensor in tensors)

        def get_pad(max_len, tensor):
            result = ()
            for i, x in enumerate(max_len[::-1]):
                result = result + (0, x - tensor.shape[-(i + 1)])
            return result

        for instance in instances:
            tensor_dict = dict(
                **{k: torch.stack(pad(tuple(r[k] for r in instance.annotation["rows"]), k))
                   for k in instance.annotation["rows"][0].keys()},
            )
            tensor_dict["window_offset"] = torch.full_like(tensor_dict["row_id"], instance.context_window.start)
            yield tensor_dict
