"""Input Formatter for multi-modal DBs with TaBERT."""

from __future__ import annotations
from dataclasses import replace
from email import header
from itertools import chain
import itertools
import logging
from collections import namedtuple
from math import ceil
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import spacy
from table_bert.table import Table
from eleet_pretrain.utils import DebugUnderlining, visualize_single
from eleet_pretrain.utils import table_from_dataframe

logger = logging.getLogger(__name__)


def first(S):
    """Return first value that is not None."""
    x = None
    for x in S:
        if x is not None:
            return x
    return None


Window = namedtuple("Window", ("start", "end"))
Answer = namedtuple("Answer", ("col_id", "query_id", "start", "end", "normed_token_ids",
                               "dependency_query_id", "dependency_answer_start"))
Mention = namedtuple("Mention", ("col_id", "start", "end"))
Instance = namedtuple("Instance", ("annotation", "separate_text", "table_id", "table_name",
                                   "row_ids", "col_ids", "text_ids", "context_window"))


Token = str
MAX_LEN_TBL_NAME = 150
MAX_DEDUPLICATION_SAMPLES = 10

class Instances(list):
    """A list of instances."""

    @property
    def annotations(self):
        """List of annotations of all instances."""
        return [instance.annotation for instance in self]

    @property
    def context_windows(self):
        """List of all context windows if the instances."""
        return [instance.context_window for instance in self]

    @property
    def row_indexes(self):
        """List of row indexes of the instances."""
        return [instance.row_index for instance in self]

    @property
    def separate_texts(self):
        """List of row indexes of the instances."""
        return [{"tokens": ["[CLS]"] + instance.separate_text,
                 "segment_a_length": len(instance.separate_text) + 1} for instance in self]



class BaseEleetInputFormatter():
    """Input Formatter for multi-modal DBs with TaBERT."""

    def __init__(self, config):
        """Set maximal length of text in transformer input."""
        self.max_sequence_len = config.max_sequence_len
        self.max_len_evidence = config.max_len_evidence
        self.max_num_answers = config.max_num_answers
        self.max_len_answer = config.max_len_answer
        self.window_overlap = config.window_overlap
        self.num_rows = config.num_vertical_layers
        self.separate_evidence_from_text = config.separate_evidence_from_text
        self.rng = np.random.default_rng(42)
        self.nlp_model = spacy.load("en_core_web_sm")
        # self.nlp_model = spacy.lang.en.English()
        self.nlp_model.add_pipe("sentencizer")

    def get_tensor_dict(self, instances, subdir="", filter_out_non_answers=True):
        """Get data for finetuning TaBERT."""
        tensor_dicts = list()
        for instances_tbl in instances:
            tensor_dict = self.to_tensor_dict(instances_tbl.annotations)
            tensor_dict["token_type_ids"] = tensor_dict["segment_ids"]
            del tensor_dict["segment_ids"]
            tensor_dict = self.handle_separate_text(instances_tbl, tensor_dict)

            array = np.array([[inst.table_id, inst.table_name, subdir] for inst in instances_tbl],
                             dtype=f"|S{MAX_LEN_TBL_NAME}")
            tensor_dict["origin"] = array
            text_ids = np.array([[(inst.text_ids[i] if i < len(inst.text_ids) else -1)
                                  for i in range(tensor_dict["input_ids"].shape[1])] for inst in instances_tbl],
                                dtype=f"|S{MAX_LEN_TBL_NAME}")
            tensor_dict["text_ids"] = text_ids

            tensor_dicts.append(tensor_dict)
        if len(tensor_dicts) == 0:
            return {}
        tensor_dict = self.merge_tensor_dicts(tensor_dicts, filter_out_non_answers=filter_out_non_answers)
        if logging.root.level <= logging.DEBUG:
            self.visualize_tensor_dict(tensor_dict)
        return tensor_dict

    def handle_separate_text(self, instances, tensor_dict):
        if self.separate_evidence_from_text:
            x = self.to_tensor_dict(instances.separate_texts)  # TODO
            x["token_type_ids"] = x["segment_ids"]
            del x["segment_ids"]
            tensor_dict = {**x, **{f"separated_evidence_{k}": v for k, v in tensor_dict.items()}}
        return tensor_dict

    def merge_tensor_dicts(self, tensor_dicts, filter_out_non_answers=True):
        mask = torch.hstack([d["has_answer"] for d in tensor_dicts])
        if not filter_out_non_answers:
            mask[:] = True
        result = dict()
        for k in tensor_dicts[0].keys():
            if k in ("sample_size", "has_answer", "has_mention"):
                continue
            
            if k in ("origin", "text_ids"):
                result[k] = np.vstack([d[k] for d in tensor_dicts])[np.array(mask)]
            else:
                result[k] = torch.vstack([d[k] for d in tensor_dicts])[mask]
        return result
    
    def get_evidence_length(self, table):
        ev_len = max(self._get_evidence_length(table.header, row) for row in table.data)
        mask_pos = max(next(chain((i for i in reversed(range(len(r))) if r[i] == ['[MASK]']), [-1]))
                       for r in table.data)
        min_ev_len = max(self._get_evidence_length(table.header[:mask_pos + 1], row[:mask_pos + 1])
                         for row in table.data)
        return ev_len, min_ev_len

    def _get_evidence_length(self, header, row_data):
        """Get the number of tokens for encoding the evidence."""
        result = 0
        for column, value_tokens in zip(header, row_data):
            truncated_value_tokens = value_tokens[:self.config.max_cell_len]

            column_input_tokens, _ = self.get_cell_input(
                column,
                truncated_value_tokens,
                token_offset=0
            )
            column_input_tokens.append(self.config.column_delimiter)
            result += len(column_input_tokens)
        return result

    def get_instances_preprocessing(self, tables, column_ids, queries, answers, mentions, texts, header_queries,
                                    header_meta, evidence_mode, skip_table_re):
        """Transform evidence-tuples, query attribute and text to tokens."""
        for table, tbl_col_ids, tbl_queries, tbl_answers, tbl_mentions, tbl_texts, tbl_header_queries, tbl_meta in zip(
                tables, column_ids, queries, answers, mentions, texts, header_queries, header_meta):
            if skip_table_re and tbl_meta["table_name"] and skip_table_re.match(tbl_meta["table_name"]):
                logger.debug(f"Skipping table: {tbl_meta['table_name']}")
                continue
            logger.debug(f"Formatting table: {tbl_meta['table_name']}")

            non_none_cols = np.array([i for i, c in enumerate(table.columns) if c is not None])
            table = table.iloc[:, non_none_cols]  # remove None cols
            tbl_col_ids = tbl_col_ids.iloc[non_none_cols]

            table, tbl_col_ids, tbl_mentions, tbl_header_queries = self.apply_evidence_mode(
                table, tbl_col_ids, tbl_queries, evidence_mode, tbl_mentions, tbl_header_queries)
            row_index = list(table.index)
            table_tokenized = table_from_dataframe(table.columns.name, table, self.nlp_model).tokenize(self.tokenizer)
            num_cell_tokens = [[len(cell) for cell in row] for row in table_tokenized.data]

            # masking of cells must be done first, because masking reduces number of tokens of table
            table_tokenized, num_cell_tokens_masked, num_cells_masked = self._mask_queries(table_tokenized, row_index,
                                                                                           tbl_col_ids, tbl_queries,
                                                                                           tbl_answers)
            info = {
                "num_cell_tokens_masked": num_cell_tokens_masked,
                "num_cell_tokens": num_cell_tokens,
                "num_cells_masked": num_cells_masked
            }
            yield self.get_instances_for_table(col_ids=tbl_col_ids,
                                               queries=tbl_queries,
                                               answers=tbl_answers,
                                               mentions=tbl_mentions,
                                               header_queries=tbl_header_queries,
                                               texts=tbl_texts,
                                               row_index=row_index,
                                               table_tokenized=table_tokenized,
                                               table_name=tbl_meta["table_name"],
                                               info=info)

    def get_instances_for_table(self, col_ids, queries, answers, mentions, header_queries, texts,
                                row_index, table_tokenized, table_name, info):
        text_tokenized = [self.tokenizer.tokenize(x) for x in texts["text"]]
        texts_encoding = self.tokenizer(texts["text"].tolist())
        col_index = list(col_ids)
        text_index = np.array(texts["text_id"])
        table_instances = []
        for window in self.cut_windows(text_tokenized, table_tokenized):
            window_input = self.get_input(text_tokenized, window, table_tokenized)
            row_instances = window_input["rows"]

            self.add_cell_labels(rows=row_instances, row_ids=row_index, col_ids=col_index, answers=answers,
                                 queries=queries, texts=texts, texts_encoding=texts_encoding, window=window)
            if "relevant_text_start" in texts.columns:
                self.add_relevant_text_labels(rows=row_instances, row_ids=row_index, texts=texts,
                                              texts_encoding=texts_encoding, window=window)
            if mentions is not None:
                self.add_mentions(rows=row_instances, row_ids=row_index, col_ids=col_index, mentions=mentions,
                                  texts=texts, texts_encoding=texts_encoding, window=window)
            if header_queries is not None:
                self.add_header_queries(rows=row_instances, row_ids=row_index, col_ids=col_index,
                                        header_queries=header_queries, texts=texts, texts_encoding=texts_encoding,
                                        window=window)
                self.add_deduplication_labels(rows=row_instances, row_ids=row_index, header_queries=header_queries,
                                              texts=texts, texts_encoding=texts_encoding, window=window)
            # self.add_text_aligns(rows=row_instances, row_ids=row_index, col_ids=col_index, aligns=aligns,
            #                      texts=texts, texts_encoding=texts_encoding, window=window)

            num_maskable_columns = min(len(row_inst['column_spans']) for row_inst in window_input["rows"])
            window_input["table_size"] = ( (len(row_instances)), num_maskable_columns)
            window_input["info"] = info
            window_instance = Instance(window_input, None, table_tokenized.id, table_name, row_index,
                                       col_index, text_index, window)
            table_instances.append(window_instance)
        return Instances(table_instances)

    def _mask_queries(self, table_tokenized, row_index, column_ids, queries, answers):
        num_cells_masked = {r: 0 for r in row_index}
        num_cell_tokens_masked = {r: 0 for r in row_index}
        row_order = row_index
        row_index = {r: i for i, r in enumerate(row_index)}
        col_index = {c: i for i, c in enumerate(column_ids)}
        for (row_id, col_id, _), _ in queries.iterrows():
            num_cells_masked[row_id] += 1
            try: 
                answer = ",".join(set(answers.loc[(row_id, col_id), "answer_surfaceform"]))
                answer_tokenized = self.tokenizer.tokenize(answer)
            except KeyError:
                answer_tokenized = []
            num_cell_tokens_masked[row_id] += len(answer_tokenized)
            table_tokenized.data[row_index[row_id]][col_index[col_id]] = ["[MASK]"]
        return table_tokenized, [num_cell_tokens_masked[r] for r in row_order], [num_cells_masked[r] for r in row_order]

    def cut_windows(self, context: List[List[Token]], table: Table) -> Iterator[Tuple[Window, Dict[str, Any]]]:
        """Cut the context into overlapping windows."""
        len_evidence, min_len_evidence = self.get_evidence_length(table)
        max_len_text = self.max_sequence_len - 1  # CLS token
        if not self.separate_evidence_from_text:
            max_len_text -= max(min_len_evidence, min(self.max_len_evidence, len_evidence)) + 2  # SEP Token

        len_longest_context = max(len(c) for c in context)
        num_windows = max(1, 1 + ceil((len_longest_context - max_len_text) / (max_len_text - self.window_overlap)))
        for i in range(num_windows):
            window_start = (max_len_text - self.window_overlap) * i
            window_end = min(window_start + max_len_text, len_longest_context)
            window = Window(window_start, window_end)

            yield window

    def visualize_tensor_dict(self, tensor_dict, num_examples=1000):
        """Visualize the tensor dict for debug purposes."""
        num_available = tensor_dict["input_ids"].shape[0]
        for i in sorted(self.rng.choice(num_available, min(num_examples, num_available), replace=False)):
            underlinings = [
                    DebugUnderlining("Masked Cells", "M", tensor_dict["answer_start"][i],
                                     tensor_dict["answer_end"][i], tensor_dict["answer_col_ids"][i],
                                     tensor_dict["normalized_answers"][i])
            ]

            if "mention_start" in tensor_dict:
                underlinings.append(
                    DebugUnderlining("Overlap", "O", tensor_dict["mention_start"][i], tensor_dict["mention_end"][i],
                                     tensor_dict["mention_col_ids"][i]))
            if "header_query_start" in tensor_dict:
                underlinings.append(
                    DebugUnderlining("Header Align", "H", tensor_dict["header_query_start"][i],
                                     tensor_dict["header_query_end"][i], tensor_dict["header_query_col_ids"][i])
                )
            if "relevant_text" in tensor_dict:
                underlinings.append(
                    DebugUnderlining("Relevant", "R", tensor_dict["relevant_text"][i, :, 0],
                                     tensor_dict["relevant_text"][i, :, 1], None))

            visualize_single(
                tokenizer=self.tokenizer,
                input_ids=tensor_dict["input_ids"][i],
                token_type_ids=tensor_dict["token_type_ids"][i],
                sequence_mask=tensor_dict["sequence_mask"][i],
                underlinings=underlinings,
                print_func=logger.debug)
            logger.debug("")

    def add_relevant_text_labels(self, rows, row_ids, texts, texts_encoding, window):
        """Add the token positions to the encodings (and the normalized answers)."""
        iterator = zip(rows, row_ids, texts.iterrows(), texts_encoding.encodings)
        for row, row_id, (_, (_, text, relevant_text_start, relevant_text_end)), row_text_enc in iterator:
            row["relevant_text_start"], row["relevant_text_end"] = self.to_token_pos(
                text, row_text_enc, relevant_text_start, relevant_text_end, window, out_of_window="window_boundary"
            )
            row["relevant_text_start"] -= window.start
            row["relevant_text_end"] -= window.start

    def add_cell_labels(self, rows, row_ids, col_ids, answers, queries, texts, texts_encoding, window):
        """Add the token positions to the encodings (and the normalized answers)."""
        iterator = zip(rows, row_ids, texts["text"].iteritems(), texts_encoding.encodings)
        for row, row_id, (_, text), row_text_enc in iterator:
            row["masked_cell_start_answer"] = []
            row["masked_cell_end_answer"] = []
            row["masked_cell_column_ids"] = []
            row["masked_cell_query_ids"] = []
            row["masked_cell_mask_token_positions"] = []
            row["masked_cell_normalized_ids"] = []
            row["dependency_query_ids"] = []
            row["dependency_answers_start"] = []

            try:
                row_answers = answers.loc[row_id]
                row_queries = queries.loc[row_id]
            except KeyError:
                continue
            answer_iterator = self.get_answers(row_answers, row_queries, text, row_text_enc, window)
            for col_id, q_id, start, end, normed_ids, dep_qid, dep_pos in answer_iterator:
                cid = col_ids.index(col_id)
                if cid >= len(row["column_spans"]):
                    continue
                row["masked_cell_start_answer"].append(start)
                row["masked_cell_end_answer"].append(end)
                row["masked_cell_column_ids"].append(cid)
                row["masked_cell_query_ids"].append(q_id)
                row["masked_cell_mask_token_positions"].append(row["column_spans"][cid]["value"][0])
                row["masked_cell_normalized_ids"] .append(normed_ids)
                row["dependency_query_ids"].append(dep_qid)
                row["dependency_answers_start"].append(dep_pos)

    def add_header_queries(self, rows, row_ids, col_ids, header_queries, texts, texts_encoding, window):
        """Add the token positions to the encodings (and the normalized answers)."""
        self.add_mentions(rows, row_ids, col_ids, header_queries[["answer_start", "answer_end"]],
                          texts, texts_encoding, window,
                          start_name="header_query_start", end_name="header_query_end",
                          column_ids_name="header_query_column_ids")

    def add_deduplication_labels(self, rows, row_ids, header_queries, texts, texts_encoding, window):
        """Add labels for deduplication."""

        for i, row_id in enumerate(row_ids):
            rows[i]["deduplication_same"] = []
            rows[i]["deduplication_diff"] = []
            try:
                row_header_queries = header_queries.loc[row_id]
            except KeyError:
                continue
            same, diff = self.get_deduplication_pairs(row_header_queries, texts.loc[row_id]["text"],
                                                      texts_encoding[i], window)
            same, diff = self.sample_deduplication_pairs(same, diff, rows[i]["tokens"])
            rows[i]["deduplication_same"] = same
            rows[i]["deduplication_diff"] = diff

    def get_deduplication_pairs(self, header_queries, text, text_encoding, window):
        header_queries = header_queries.sort_index()
        same = set()
        diff = set()
        for col_id in header_queries.index.unique("col_id"):
            answers = header_queries.loc[col_id].reset_index().set_index("answer_numeric_id")
            answer_ids = answers.index.unique()
            for a1, a2 in filter(lambda x: x[0] <= x[1], itertools.product(answer_ids, answer_ids)):
                for l1 in answers.loc[[a1]].itertuples():
                    l1_tok = self.to_token_pos(text, text_encoding,
                                               l1.answer_start, l1.answer_end, window)
                    if l1_tok == (False, False):
                        continue
                    for l2 in answers.loc[[a2]].itertuples():
                        l2_tok = self.to_token_pos(text, text_encoding,
                                                   l2.answer_start, l2.answer_end, window)
                        if l2_tok == (False, False) or l1_tok[0] >= l2_tok[0]:
                            continue
                        to_add = tuple(tuple(e - window.start for e in t) for t in (l1_tok, l2_tok))
                        (same if a1 == a2 else diff).add(to_add)
        conflict = same & diff
        same -= conflict
        diff -= conflict
        return same, diff

    def sample_deduplication_pairs(self, same, diff, tokens):
        same_len_diff = sorted({(x, y) for x, y in same if tokens[x[0]: x[1]] != tokens[y[0]: y[1]]})
        sample_size_diff = min(MAX_DEDUPLICATION_SAMPLES, len(same_len_diff))
        result_same = self.rng.choice(same_len_diff, size=sample_size_diff, replace=False).tolist()
        diff = sorted(diff)
        result_diff = self.rng.choice(diff, size=min(len(diff), MAX_DEDUPLICATION_SAMPLES),
                                      replace=False).tolist()
        return result_same, result_diff

    def add_mentions(self, rows, row_ids, col_ids, mentions, texts, texts_encoding, window,
                     start_name="overlap_start_mention", end_name="overlap_end_mention",
                     column_ids_name="overlap_column_ids"):
        """Add the token positions to the encodings (and the normalized answers)."""
        iterator = zip(rows, row_ids, texts["text"].iteritems(), texts_encoding.encodings)
        for row, row_id, (_, text), row_text_enc in iterator:
            row[start_name] = []
            row[end_name] = []
            row[column_ids_name] = []

            try:
                row_mentions = mentions.loc[row_id]
            except KeyError:
                continue
            for col_id, start, end in self.get_mentions(row_mentions, text, row_text_enc, window):
                cid = col_ids.index(col_id)
                if cid >= len(row["column_spans"]):
                    continue
                row[start_name].append(start)
                row[end_name].append(end)
                row[column_ids_name].append(cid)

    def get_mentions(self, answers, text, text_enc, window):
        already_done_dict = dict()
        for (col_id, _), (start, end) in answers.sort_values(answers.columns[0]).iterrows():
            already_done = already_done_dict[col_id] = already_done_dict.get(col_id, set())
            token_start, token_end = self.to_token_pos(text, text_enc, start, end, window, already_done)
            if token_start is not False and token_end is not False:
                yield Mention(col_id, token_start - window.start, token_end - window.start)

    def get_answers(self, answers, queries, text, text_enc, window):
        """Get the answers for the given row and window."""
        already_done_dict = dict()
        for (col_id, q_id, _), (start, end, _, normed, _) in answers.sort_values("answer_start").iterrows():
            dependency_query_id, dependency_answer_start = queries.loc[col_id, q_id]
            already_done = already_done_dict[col_id, q_id] = already_done_dict.get((col_id, q_id), set())
            token_start, token_end = self.to_token_pos(text, text_enc, start, end, window, already_done)
            if dependency_answer_start >= 0:
                dependency_answer_start = self.to_token_pos(text, text_enc, dependency_answer_start,
                                                            dependency_answer_start + 1, window,
                                                            out_of_window="cross_reference")[0]
            if token_start is not False and token_end is not False:
                normed_ids = self.tokenizer(normed)["input_ids"]
                yield Answer(col_id, q_id, token_start - window.start, token_end - window.start, normed_ids,
                             dependency_query_id, dependency_answer_start)

    def to_token_pos(self, text, text_enc, start, end, window, already_done=None, out_of_window="skip"):
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
        if already_done is not None and (
                (token_start, token_end) in already_done or len(already_done) >= self.max_num_answers):
            return False, False

        if already_done is not None:
            already_done.add((token_start, token_end))
        return token_start, token_end

    def num_mask_tokens(self, i, answer):
        raise NotImplementedError

    def apply_evidence_mode(self, table, col_ids, queries, evidence_mode, mentions, header_queries):
        if evidence_mode == "full":
            return table, col_ids, mentions, header_queries

        # identify not queried columns
        cols_to_delete = set(col_ids) - set(queries.index.unique("col_id"))
        cols_others_depend_on = set([c.split("-")[0] for c in col_ids if c is not None and "-" in c])
        if evidence_mode == "row-id":
            cols_others_depend_on.add("id")
        cols_to_delete -= cols_others_depend_on

        # delete not queried columns
        header_queries = header_queries.drop(list(cols_to_delete), level="col_id", errors="ignore").sort_index()
        mentions = mentions.drop(list(cols_to_delete), level="col_id", errors="ignore").sort_index()
        keep_i = [i for i, c in enumerate(col_ids) if c not in cols_to_delete]
        table = table.iloc[:, keep_i]
        col_ids = col_ids.iloc[keep_i]

        # set non queried cells to none
        all_cells = set(itertools.product(table.index, col_ids))
        queried_cells = set(map(lambda x: tuple(x[1]), queries.index.to_frame()[["row_id", "col_id"]].iterrows()))
        nullify = all_cells - queried_cells
        for row_id, col_id in nullify:
            if col_id in cols_others_depend_on:
                continue
            table.loc[row_id][np.array(col_ids == col_id)] = None
            header_queries = header_queries.drop((row_id, col_id), errors="ignore")
            mentions = mentions.drop((row_id, col_id), errors="ignore")
        
        return table, col_ids, mentions.sort_index(), header_queries.sort_index()