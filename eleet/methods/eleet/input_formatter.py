from collections import namedtuple
from itertools import chain
from math import ceil
from typing import Any, Dict, Iterator, List, Tuple
from attrs import define, field
import numpy as np
import pandas as pd
import spacy
import table_bert.table
import torch
from table_bert.vertical.input_formatter import VerticalAttentionTableBertInputFormatter
from eleet_pretrain.model.config import VerticalEleetConfig
from eleet_pretrain.utils import table_from_dataframe
from transformers import BertTokenizerFast
import logging

logger = logging.getLogger(__name__)


Token = str

Window = namedtuple("Window", ("start", "end"))
Answer = namedtuple("Answer", ("col_id", "query_id", "start", "end", "normed_token_ids",
                               "dependency_query_id", "dependency_answer_start"))
Mention = namedtuple("Mention", ("col_id", "start", "end"))
Instance = namedtuple("Instance", ("annotation", "separate_text", "table_id", "table_name",
                                   "row_ids", "col_ids", "context_window"))


class ELEETInputFormatter(VerticalAttentionTableBertInputFormatter):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.nlp_model = spacy.load("en_core_web_sm")
        self.nlp_model.add_pipe("sentencizer")
        super().__init__(self.config, self.tokenizer)

    def get_instances(self, queries: pd.DataFrame, table: pd.DataFrame, texts: pd.DataFrame,
                      table_name: str, row_ids: List[int], header_queries: pd.DataFrame, answers: pd.DataFrame):
        table_tokenized = table_from_dataframe(table_name, table.astype(str), self.nlp_model).tokenize(self.tokenizer)
        col_ids = pd.Series(table.columns)

        # masking of cells must be done first, because masking reduces number of tokens of table
        table_tokenized = self._mask_queries(table_tokenized, queries)
        yield from  self.get_instances_for_table(col_ids=col_ids,
                                                 queries=queries,
                                                 answers=answers,
                                                 mentions=None,  # TODO
                                                 header_queries=header_queries,
                                                 texts=texts,
                                                 row_index=row_ids,
                                                 table_tokenized=table_tokenized,
                                                 table_name=table_name)

    def _mask_queries(self, table_tokenized, queries):
        for _, (row_id, col_id, _) in queries.iterrows():
            table_tokenized.data[row_id][col_id] = ["[MASK]"]
        return table_tokenized

    def get_instances_for_table(self, col_ids, queries, answers, mentions, header_queries, texts,
                                row_index, table_tokenized, table_name):
        texts_encoding = self.tokenizer(texts.tolist(), return_tensors="pt", padding=True)
        table_encoding = self.get_table_encoding(table_tokenized)
        id_col_values = [row[0] for row in table_tokenized.data]

        col_index = list(col_ids)
        for window in self.cut_windows(texts_encoding, table_encoding):
            window_input = self.get_input(texts_encoding, window, table_encoding, queries, row_index,
                                          table_name, id_col_values)
            row_instances = window_input["rows"]

            if answers is not None:
                self.add_query_labels(rows=row_instances, answers=answers,
                                      texts_encoding=texts_encoding, texts=texts, window=window)
            if header_queries is not None:
                self.add_query_labels(rows=row_instances, answers=header_queries,
                                      texts_encoding=texts_encoding, texts=texts, window=window,
                                      label_name="header_query_labels")
                self.add_deduplication_labels(rows=row_instances, header_queries=header_queries, col_ids=col_ids,
                                              texts=texts, texts_encoding=texts_encoding, window=window)

            num_columns = min(row_inst['token_type_ids'].max() for row_inst in window_input["rows"]).item()
            window_input["table_size"] = (len(row_instances), num_columns)
            window_instance = Instance(window_input, None, table_tokenized.id, table_name, row_index,
                                       col_index, window)
            yield window_instance


    def cut_windows(self, texts_encoding, table_encoding):
        """Cut the context into overlapping windows."""
        len_evidence, min_len_evidence = self.get_evidence_length(table_encoding)
        max_len_text = self.config.max_sequence_len - 2 - self.config.max_cell_len + 1 # SEP, CLS, buffer
        max_len_text -= max(min_len_evidence, min(self.config.max_len_evidence, len_evidence))
        len_longest_context = max(len(c) for c in texts_encoding.input_ids) - 2 # SEP, CLS
        num_windows = max(1, 1 + ceil((len_longest_context - max_len_text)
                                      / (max_len_text - self.config.window_overlap)))
        for i in range(num_windows):
            window_start = (max_len_text - self.config.window_overlap) * i
            window_end = min(window_start + max_len_text, len_longest_context)
            window = Window(window_start, window_end)

            yield window

    def get_evidence_length(self, table_encoding):
        ev_len = max(map(len, table_encoding))
        last_mask_pos = max(map(lambda li: len(li) - 1 - li[::-1].index('[MASK]'),
                                [x for x in table_encoding if "[MASK]" in x]))
        min_ev_len = last_mask_pos + 2 # SEP TOKEN
        return ev_len, min_ev_len

    def get_table_encoding(self, table_tokenized):
        """Get the number of tokens for encoding the evidence."""
        result = []
        for row in table_tokenized.data:
            row_result = []
            for column, value_tokens in zip(table_tokenized.header, row):
                truncated_value_tokens = value_tokens[:self.config.max_cell_len]

                column_input_tokens, _ = self.get_cell_input(
                    column,
                    truncated_value_tokens,
                    token_offset=0
                )
                column_input_tokens.append(self.config.column_delimiter)
                row_result += column_input_tokens
            result.append(row_result)
        return result

    def get_input(self, texts_encoding, window, table_encoding, queries, row_index, table_name, id_col_values):
        row_instances = []
        window_slice = slice(1 + window.start, 1 + window.end)
        window_input_ids = texts_encoding["input_ids"][:, window_slice]
        window_attention_mask = texts_encoding["attention_mask"][:, window_slice]
        queries = queries.set_index("row_id")

        for i, row_id in zip(range(len(table_encoding)), row_index):
            row_input_ids, row_attention_mask = self.window_special_tokens(window_input_ids[i],
                                                                           window_attention_mask[i])
            row_queries = queries.loc[[i]] if i in queries.index else queries.loc[[]]
            window_text = self.get_input_row(table_encoding[i], row_input_ids, 
                                             row_attention_mask, row_queries)
            window_text["row_id"] = torch.tensor([row_id], dtype=int)
            window_text["col_offset"] = torch.tensor([int(table_name.split("-")[-1])], dtype=int)
            window_text["id_col_value"] = torch.tensor(self.tokenizer.convert_tokens_to_ids(id_col_values[i]),
                                                       dtype=int)
            window_text["id_col_vec"] = torch.zeros(self.config.hidden_size, dtype=float)
            row_instances.append(window_text)
        result = {"rows": row_instances}
        return result

    def window_special_tokens(self, row_input_ids, row_attention_mask):
        append_sep = (row_input_ids[-1] != self.tokenizer.sep_token_id) & (row_attention_mask[-1] > 0)
        row_input_ids = torch.hstack((torch.tensor([self.tokenizer.cls_token_id]),
                                          row_input_ids,
                                          torch.tensor([self.tokenizer.sep_token_id if append_sep else 0])))
        row_attention_mask = torch.hstack((torch.tensor([1]),
                                           row_attention_mask,
                                           torch.tensor([1 if append_sep else 0])))
        return row_input_ids, row_attention_mask

    def get_input_row(self, row_encoding, input_ids, attention_mask, queries):
        num_cols = row_encoding.count("[SEP]")
        row_token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(row_encoding))
        col_ids = torch.cumsum(row_token_ids == self.tokenizer.sep_token_id, 0)
        col_ids[row_token_ids == self.tokenizer.sep_token_id] = -1
        queries_dict = dict(x[1] for x in queries.iterrows())
        query_mask = torch.full((num_cols,), -1, dtype=int)
        for i in range(0, num_cols):
            query_mask[i] = queries_dict.get(i, -1)


        mask = attention_mask.bool()
        num_active = mask.sum()
        window_text = {
            "input_ids": torch.hstack((input_ids[mask], row_token_ids)),
            "token_type_ids": torch.hstack((torch.zeros(num_active, dtype=int), torch.ones_like(row_token_ids))),
            "sequence_mask": torch.hstack((attention_mask[mask], torch.ones_like(row_token_ids))),
            "column_token_position_to_column_ids": torch.hstack((torch.full((num_active,), -1, dtype=int), col_ids)),
            "context_token_positions": torch.hstack((torch.arange(num_active - 1, dtype=int),
                                                    torch.zeros(row_token_ids.shape[0] + 1, dtype=int))),
            "context_token_mask": torch.hstack((torch.ones(num_active - 1, dtype=int),
                                                torch.zeros(row_token_ids.shape[0] + 1, dtype=int))),
            "table_mask": torch.ones(num_cols, dtype=int),
            "query_mask": query_mask
        }
        return window_text

    def add_query_labels(self, rows, answers, texts_encoding, texts, window, label_name="query_labels"):
        answers = answers.set_index("row_id")
        for i, row in enumerate(rows):
            row[label_name] = torch.zeros(self.config.max_num_cols, self.config.max_sequence_len, dtype=int)
            if i not in answers.index:
                continue

            for _, answer in answers.loc[[i]].iterrows():
                token_start, token_end = self.to_token_pos(text=texts[i], text_enc=texts_encoding[i],
                                                           start=answer["start"], end=answer["end"], window=window)
                if token_start is False or token_end is False:
                    continue
                if row[label_name][answer["col_id"]][token_start] != 0:
                    continue
                row[label_name][answer["col_id"]][token_start: token_end] = 1
                row[label_name][answer["col_id"]][token_start] = 2
    
    def add_deduplication_labels(self, rows, header_queries, col_ids, texts, texts_encoding, window):
        answers = header_queries.set_index("row_id")
        for i, row in enumerate(rows):
            row["deduplication_labels"] = torch.zeros(
                self.config.max_num_deduplication_labels_per_row,
                2 + self.config.max_cell_len + self.config.deduplication_max_normed_len,
                dtype=int
            )
            if i not in answers.index:
                continue

            j = 0
            for _, answer in answers.loc[[i]].iterrows():
                token_start, token_end = self.to_token_pos(text=texts[i], text_enc=texts_encoding[i],
                                                           start=answer["start"], end=answer["end"], window=window)
                if token_start is False or token_end is False:
                    continue
                if j >= self.config.max_num_deduplication_labels_per_row:
                    break
                normed_tokens = self.tokenizer(
                    answer["normed"], return_tensors="pt"
                )["input_ids"][0, 1:-1][:self.config.deduplication_max_normed_len]
                col_tokens = self.tokenizer(
                    col_ids[answer["col_id"]], return_tensors="pt"
                )["input_ids"][0, 1:-1][:self.config.max_cell_len]

                row["deduplication_labels"][j, 0] = token_start
                row["deduplication_labels"][j, 1] = token_end
                row["deduplication_labels"][j, 2: 2 + len(col_tokens)] = col_tokens
                row["deduplication_labels"][j, 2 + self.config.max_cell_len:
                                            2 + self.config.max_cell_len + len(normed_tokens)] = normed_tokens
                j += 1

    def to_token_pos(self, text, text_enc, start, end, window, out_of_window="skip"):
        o, b, f = range(1), range(-1, -min(start + 1, 3), -1), range(1, 3)
        token_start = first(text_enc.char_to_token(start + k) for k in chain(o, b, f))
        token_end = first(text_enc.char_to_token(end + k - 1) for k in chain(o, f, b))
        if token_end is not None:
            token_end += 1

        window_start_cls = window.start + 1  # account for [CLS] tag ignored so far in windows
        window_end_cls = window.end + 1

        if token_start is None or token_end is None or token_start == token_end:
            logger.warn(f"Unable to compute token positions: "
                        f"{text[:start]} ---S({start}) {text[start: end]} ({end})E--- {text[end:]}")
            if out_of_window == "skip":
                return (False, False)
            elif out_of_window == "window_boundary":
                return (token_start or window_start_cls, token_end or window_end_cls)
            elif out_of_window == "cross_reference":
                return (0, 0)
            else:
                raise Exception("Unknown value for out_of_window")
        if token_end > window_end_cls or token_start < window_start_cls:
            if  out_of_window == "skip":
                return (False, False)
            elif out_of_window == "window_boundary":
                return (min(max(token_start, window_start_cls), window_end_cls),
                        min(max(token_end, window_start_cls), window_end_cls))
            elif out_of_window == "cross_reference":
                pass
            else:
                raise Exception("Unknown value for out_of_window")
        return token_start - window.start, token_end - window.start



def first(S):
    """Return first value that is not None."""
    x = None
    for x in S:
        if x is not None:
            return x
    return None