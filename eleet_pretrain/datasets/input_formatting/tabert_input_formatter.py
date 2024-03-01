"""Input Formatter for multi-modal DBs with TaBERT."""

import pandas as pd
import logging
from typing import List

import torch
from table_bert.table import Table
from table_bert.vertical.input_formatter import VerticalAttentionTableBertInputFormatter
from eleet_pretrain.model.config import BaseEleetConfig
from table_bert.vertical.dataset import collate
from eleet_pretrain.datasets.input_formatting.base_input_formatter import MAX_DEDUPLICATION_SAMPLES, BaseEleetInputFormatter
from eleet_pretrain.utils import table_from_dataframe

logger = logging.getLogger(__name__)


class EleetTaBertInputFormatter(BaseEleetInputFormatter, VerticalAttentionTableBertInputFormatter):
    def __init__(self, config: BaseEleetConfig, tokenizer):
        VerticalAttentionTableBertInputFormatter.__init__(self, config, tokenizer)
        BaseEleetInputFormatter.__init__(self, config)

    def get_instances(self, *, queries, table, texts, table_name, answers, header_queries):
        row_index = list(table.index)
        table_tokenized = table_from_dataframe(table_name, table.astype(str), self.nlp_model).tokenize(self.tokenizer)
        num_cell_tokens = [[len(cell) for cell in row] for row in table_tokenized.data]
        col_ids = pd.Series(table.columns)

        # masking of cells must be done first, because masking reduces number of tokens of table
        table_tokenized, num_cell_tokens_masked, num_cells_masked = self._mask_queries(table_tokenized, row_index,
                                                                                       col_ids, queries, answers)
        info = {
            "num_cell_tokens_masked": num_cell_tokens_masked,
            "num_cell_tokens": num_cell_tokens,
            "num_cells_masked": num_cells_masked
        }
        return self.get_instances_for_table(col_ids=col_ids,
                                            queries=queries,
                                            answers=answers,
                                            mentions=None,
                                            header_queries=header_queries,
                                            texts=texts,
                                            row_index=row_index,
                                            table_tokenized=table_tokenized,
                                            table_name=table_name,
                                            info=info)

    def to_tensor_dict(self, annotations):
        examples = []  # from VerticalTableBert.to_tensor_dict
        for annotation in annotations:
            for row_inst in annotation['rows']:
                row_inst['token_ids'] = self.tokenizer.convert_tokens_to_ids(row_inst['tokens'])
            examples.append(annotation)
        tensor_dict = collate(examples, config=self.config, train=False)

        tensor_dict = self.collate_masked_cells_prediction(examples, tensor_dict)
        tensor_dict = self.collate_info(examples, tensor_dict)
        tensor_dict = self.pad(tensor_dict)
        return tensor_dict

    def collate_info(self, examples, tensor_dict):
        batch_size = len(examples)
        max_row_num = max(len(inst["rows"]) for inst in examples)
        max_col_num = max(len(x) for inst in examples for x in inst["info"]["num_cell_tokens"])

        table_size = torch.zeros(batch_size, 2, dtype=torch.long)
        num_cell_tokens_masked = torch.zeros(batch_size, max_row_num, dtype=torch.long)
        num_cell_tokens = torch.zeros(batch_size, max_row_num, max_col_num, dtype=torch.long)
        num_cells_masked = torch.zeros(batch_size, max_row_num, dtype=torch.long)

        for i, example in enumerate(examples):
            row_indices = torch.arange(len(example["rows"]))
            table_size[i] = torch.tensor(example["table_size"], dtype=torch.long)
            num_cell_tokens_masked[i, row_indices] = torch.tensor(example["info"]["num_cell_tokens_masked"],
                                                                  dtype=torch.long)
            num_cells_masked[i, row_indices] = torch.tensor(example["info"]["num_cells_masked"], dtype=torch.long)

            for j in range(len(example['rows'])):
                col_indices = torch.arange(len(example["info"]["num_cell_tokens"][j]))
                num_cell_tokens[i, j, col_indices] = torch.tensor(example["info"]["num_cell_tokens"][j])

        tensor_dict.update({
            "table_size": table_size,
            "num_cell_tokens_masked": num_cell_tokens_masked,
            "num_cell_tokens": num_cell_tokens,
            "num_cells_masked": num_cells_masked,
        })
        return tensor_dict
    
    def collate_masked_cells_prediction(self, examples, tensor_dict):
        batch_size = len(examples)
        max_row_num = max(len(inst['rows']) for inst in examples)
        max_num_answers = max(
            len(row['masked_cell_start_answer'])
            for e in examples
            for row in e['rows']
        )
        all_keys = set(k for e in examples for row in e['rows'] for k in row.keys())
        mentions_enabled = "overlap_start_mention" in all_keys
        header_queries_enabled = "header_query_start" in all_keys
        relevant_text_enabled = "relevant_text_start" in all_keys
        deduplication_enabled = "deduplication_same" in all_keys

        if mentions_enabled:
            max_num_mentions = max(
                len(row['overlap_start_mention'])
                for e in examples
                for row in e['rows']
            )

        if header_queries_enabled:
            max_num_header_query_answers = max(
                len(row['header_query_start'])
                for e in examples
                for row in e['rows']
            )

        has_answer = torch.zeros(batch_size, dtype=torch.bool)
        answer_start = torch.zeros(batch_size, max_row_num, max_num_answers, dtype=torch.long)
        answer_end = torch.zeros(batch_size, max_row_num, max_num_answers, dtype=torch.long)
        answer_col_ids = torch.full((batch_size, max_row_num, max_num_answers), -1, dtype=torch.long)
        answer_qid = torch.full((batch_size, max_row_num, max_num_answers), -1, dtype=torch.long)
        answer_dep_qid = torch.full((batch_size, max_row_num, max_num_answers), -1, dtype=torch.long)
        answer_dep_a_start = torch.full((batch_size, max_row_num, max_num_answers), -1, dtype=torch.long)
        mask_pos = torch.zeros(batch_size, max_row_num, max_num_answers, dtype=torch.long)
        normalized_answers = torch.zeros(batch_size, max_row_num, max_num_answers, self.max_len_answer,
                                         dtype=torch.long)
        if mentions_enabled:
            mention_start = torch.zeros(batch_size, max_row_num, max_num_mentions, dtype=torch.long)
            mention_end = torch.zeros(batch_size, max_row_num, max_num_mentions, dtype=torch.long)
            mention_col_ids = torch.full((batch_size, max_row_num, max_num_mentions), -1, dtype=torch.long)

        if header_queries_enabled:
            hq_start = torch.zeros(batch_size, max_row_num, max_num_header_query_answers, dtype=torch.long)
            hq_end = torch.zeros(batch_size, max_row_num, max_num_header_query_answers, dtype=torch.long)
            hq_col_ids = torch.full((batch_size, max_row_num, max_num_header_query_answers), -1, dtype=torch.long)

        relevant_text = torch.zeros(batch_size, max_row_num, 2, dtype=torch.long)
        deduplication_labels = torch.zeros((batch_size, max_row_num, 2, MAX_DEDUPLICATION_SAMPLES, 2, 2),
                                           dtype=torch.long)

        for i, example in enumerate(examples):
            for j, row_inst in enumerate(example['rows']):
                if relevant_text_enabled:
                    relevant_text[i, j] = torch.tensor((row_inst["relevant_text_start"], row_inst["relevant_text_end"]), dtype=torch.long)

                answer_indices = torch.arange(len(row_inst["masked_cell_start_answer"]))
                answer_start[i, j, answer_indices] = torch.tensor(row_inst["masked_cell_start_answer"], dtype=torch.long)
                answer_end[i, j, answer_indices] = torch.tensor(row_inst["masked_cell_end_answer"], dtype=torch.long)
                answer_col_ids[i, j, answer_indices] = torch.tensor(row_inst["masked_cell_column_ids"], dtype=torch.long)
                answer_qid[i, j, answer_indices] = torch.tensor(row_inst["masked_cell_query_ids"], dtype=torch.long)
                answer_dep_qid[i, j, answer_indices] = torch.tensor(row_inst["dependency_query_ids"], dtype=torch.long)
                answer_dep_a_start[i, j, answer_indices] = torch.tensor(row_inst["dependency_answers_start"], dtype=torch.long)
                mask_pos[i, j, answer_indices] = torch.tensor(row_inst["masked_cell_mask_token_positions"], dtype=torch.long)
                for k in answer_indices:
                    a = row_inst["masked_cell_normalized_ids"][k][:self.max_len_answer]
                    has_answer[i] = True
                    normalized_answer_indices = torch.arange(len(a))
                    normalized_answers[i, j, k, normalized_answer_indices] = torch.tensor(a, dtype=torch.long)

                if mentions_enabled:
                    mention_indices = torch.arange(len(row_inst["overlap_start_mention"]))
                    mention_start[i, j, mention_indices] = torch.tensor(row_inst["overlap_start_mention"], dtype=torch.long)
                    mention_end[i, j, mention_indices] = torch.tensor(row_inst["overlap_end_mention"], dtype=torch.long)
                    mention_col_ids[i, j, mention_indices] = torch.tensor(row_inst["overlap_column_ids"], dtype=torch.long)

                if header_queries_enabled:
                    hq_indices = torch.arange(len(row_inst["header_query_start"]))
                    hq_start[i, j, hq_indices] = torch.tensor(row_inst["header_query_start"], dtype=torch.long)
                    hq_end[i, j, hq_indices] = torch.tensor(row_inst["header_query_end"], dtype=torch.long)
                    hq_col_ids[i, j, hq_indices] = torch.tensor(row_inst["header_query_column_ids"], dtype=torch.long)

                if deduplication_enabled:
                    deduplication_labels[i, j, 0, :len(row_inst["deduplication_same"])] = torch.tensor(
                        row_inst["deduplication_same"], dtype=torch.long).view(-1, 2, 2)
                    deduplication_labels[i, j, 1, :len(row_inst["deduplication_diff"])] = torch.tensor(
                        row_inst["deduplication_diff"], dtype=torch.long).view(-1, 2, 2)

        tensor_dict.update({
            "answer_start": answer_start, "answer_end": answer_end, "answer_col_ids": answer_col_ids,
            "mask_token_positions": mask_pos, "normalized_answers": normalized_answers, "has_answer": has_answer,
            "answer_qid": answer_qid, "answer_dep_qid": answer_dep_qid, "answer_dep_a_start": answer_dep_a_start,
        })

        if mentions_enabled:
            tensor_dict.update({
                "mention_start": mention_start, "mention_end": mention_end, "mention_col_ids": mention_col_ids,
            })
        if header_queries_enabled:
            tensor_dict.update({
                "header_query_start": hq_start, "header_query_end": hq_end, "header_query_col_ids": hq_col_ids,
            })
        if relevant_text_enabled:
            tensor_dict.update({
                "relevant_text": relevant_text,
            })
        if deduplication_enabled:
            tensor_dict.update({
                "deduplication_labels": deduplication_labels
            })

        return tensor_dict

    def pad(self, tensor_dict):   # TODO Do all these sizes make sense? Maybe reduce sizes in DataCollator
        batch_size = tensor_dict["input_ids"].shape[0]
        num_cols = self.config.max_num_cols + self.config.label_col
        num_rows = self.config.num_vertical_layers
        max_sequence_len = self.config.max_sequence_len
        max_len_answer = self.config.max_len_answer
        max_num_answers = self.config.max_num_queries * self.config.max_num_answers
        target_shape = {
            "input_ids": (batch_size, num_rows, max_sequence_len),
            "segment_ids": (batch_size, num_rows, max_sequence_len),
            "context_token_positions": (batch_size, num_rows, max_sequence_len),
            "column_token_position_to_column_ids": (batch_size, num_rows,
                                                    max_sequence_len),
            "sequence_mask": (batch_size, num_rows, max_sequence_len),
            "context_token_mask": (batch_size, num_rows, max_sequence_len),
            "table_mask": (batch_size, num_rows, num_cols),
            "answer_col_ids": (batch_size, num_rows, max_num_answers),
            "answer_end": (batch_size, num_rows, max_num_answers),
            "answer_start": (batch_size, num_rows, max_num_answers),
            "answer_qid": (batch_size, num_rows, max_num_answers),
            "answer_dep_qid": (batch_size, num_rows, max_num_answers),
            "answer_dep_a_start": (batch_size, num_rows, max_num_answers),
            "mask_token_positions": (batch_size, num_rows, max_num_answers),
            "mention_col_ids": (batch_size, num_rows, max_num_answers),
            "mention_end": (batch_size, num_rows, max_num_answers),
            "mention_start": (batch_size, num_rows, max_num_answers),
            "header_query_col_ids": (batch_size, num_rows, max_num_answers),
            "header_query_end": (batch_size, num_rows, max_num_answers),
            "header_query_start": (batch_size, num_rows, max_num_answers),
            # "align_col_ids": (batch_size, num_rows, max_num_answers),
            # "align_end": (batch_size, num_rows, max_num_answers),
            # "align_start": (batch_size, num_rows, max_num_answers),
            "normalized_answers": (batch_size, num_rows, max_num_answers,
                                   max_len_answer),
            "table_size": (batch_size, 2),
            "num_cell_tokens_masked": (batch_size, num_rows),
            "num_cell_tokens": (batch_size, num_rows, num_cols),
            "num_cells_masked": (batch_size, num_rows),
            # "num_context_tokens_masked": (batch_size, num_rows),  TODO move
            "relevant_text": (batch_size, num_rows, 2),
            "deduplication_labels": (batch_size, num_rows, 2, MAX_DEDUPLICATION_SAMPLES, 2, 2)
        }
        fill_value = {
            # "masked_context_token_labels": -1,  TODO move
            "masked_column_token_labels": -1,
            "column_token_position_to_column_ids": -1,
            "answer_col_ids": -1,
            "header_query_col_ids": -1,
            # "align_col_ids": -1,
            "mention_col_ids": -1,
            "answer_qid": -1,
            "answer_dep_qid": -1,
            "answer_dep_a_start": -1,
        }
        reset_key = "column_token_position_to_column_ids"
        tensor_dict[reset_key][tensor_dict[reset_key] == tensor_dict[reset_key].max()] = -1
        for k, shape in target_shape.items():
            if k not in tensor_dict:
                continue
            tensor = torch.full(shape, fill_value.get(k, 0))
            truncated_shape = tuple(min(t, s) for t, s in zip(shape, tensor_dict[k].shape))
            truncated_selector = tuple(slice(0, x) for x in truncated_shape)
            if any(t < s for t, s in zip(shape, tensor_dict[k].shape)):
                logger.warn(f"{k}-tensor of shape {tensor_dict[k].shape} is truncated to shape {truncated_shape}.")

            tensor[truncated_selector] = tensor_dict[k][truncated_selector]
            tensor_dict[k] = tensor
        return tensor_dict
    
    def num_mask_tokens(self, i, answer):
        return len(self.answer_tokens(i, answer))

    def get_input(self, context: List[str], window, table: Table):
        row_instances = []

        for row_data, row_context in zip(table.data, context):
            if isinstance(row_data, dict):
                row_data = [row_data[col.name] for col in table.header]
            row_instance = self.get_row_input(row_context[window.start: window.end], table.header, row_data)
            row_instances.append(row_instance)

        result = {"rows": row_instances}
        return result