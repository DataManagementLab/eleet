from functools import partial
import torch
import logging
import inspect
from typing import Optional
from dataclasses import dataclass

from eleet_pretrain.model.base_model import BaseEleetModel
from transformers.file_utils import ModelOutput
from eleet_pretrain.model.table_decoding_dependencies import get_answer_end
from eleet_pretrain.utils import compute_span_similarities, to_iob
from torch import nn


MAX_NUM_SPAN_EMBEDDINGS = 50
logger = logging.getLogger(__name__)

@dataclass
class ELEETInferenceModelOutput(ModelOutput):
    """Model output for ELEET transformers."""
    context_encoding: Optional[torch.FloatTensor] = None
    schema_encoding: Optional[torch.FloatTensor] = None
    table_encoding: Optional[torch.FloatTensor] = None
    deduplication_context_encoding: Optional[torch.FloatTensor] = None


@dataclass
class ELEETFinetuningModelOutput(ModelOutput):
    """Model output for ELEET transformers."""
    loss: Optional[torch.FloatTensor] = None
    sd_loss_value: Optional[torch.FloatTensor] = None
    hq_loss_value: Optional[torch.FloatTensor] = None
    dup_loss_value: Optional[torch.FloatTensor] = None
    sd_f1: Optional[torch.FloatTensor] = None
    hq_f1: Optional[torch.FloatTensor] = None
    dup_f1: Optional[torch.FloatTensor] = None


class ELEETInferenceModel(BaseEleetModel):
    def __init__(self, *args, union_mode=False, fast_mode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.union_mode = union_mode
        self.fast_mode = fast_mode
        self.index_mode = False

    def set_union_mode(self, union_mode=True):
        self.union_mode = union_mode

    def set_index_mode(self, index_mode=True):
        self.index_mode = index_mode

    def set_fast_mode(self, fast_mode=True):
        assert not fast_mode or self.union_mode
        self.fast_mode = fast_mode

    def forward(self,
                column_token_position_to_column_ids,
                context_token_mask,
                context_token_positions,
                input_ids,
                sequence_mask,
                table_mask,
                query_mask,
                token_type_ids,
                **kwargs):
        assert not self.training
        context_encoding, schema_encoding, table_encoding, hidden_states = self.encode(
            column_token_position_to_column_ids=column_token_position_to_column_ids,
            context_token_mask=context_token_mask,
            context_token_positions=context_token_positions,
            input_ids=input_ids,
            sequence_mask=sequence_mask,
            table_mask=table_mask,
            token_type_ids=token_type_ids
        )
        deduplication_context_encoding = self.deduplication_encode(
            hidden_states[-2],
            column_token_position_to_column_ids=column_token_position_to_column_ids,
            context_token_mask=context_token_mask,
            context_token_positions=context_token_positions,
            input_ids=input_ids,
            sequence_mask=sequence_mask,
            table_mask=table_mask,
            token_type_ids=token_type_ids
        )
        # embeddings = self.get_span_embeddings(context_encoding=deduplication_context_encoding, sd_pred=sd_pred,
        #                                       **kwargs)

        return ELEETInferenceModelOutput(
            context_encoding=context_encoding,
            schema_encoding=schema_encoding,
            table_encoding=table_encoding,
            deduplication_context_encoding=deduplication_context_encoding
        )

    def encode_flatten(self, *args, batch_size, max_row_num, mask):
        flattened = super().encode_flatten(*args, batch_size=batch_size, max_row_num=max_row_num, mask=mask)
        if not self.fast_mode or self.training:
            return flattened
        fast_mask = torch.zeros(batch_size * max_row_num, device=mask.device, dtype=bool)
        fast_mask[:max_row_num] = mask[0]
        fast_mask[mask.sum(1) + torch.arange(0, batch_size * max_row_num, max_row_num, device=mask.device) - 1] = 1
        return tuple(f[fast_mask] for f in flattened[:-1]) + (mask, )

    def flatten_adjust(self, arg, idx1, idx2):
        return arg[idx1, idx2].reshape(arg.shape)

    def encode_unflatten(self, *args, batch_size, max_row_num, sequence_len, flatten_info):
        adjust_func = None
        if self.fast_mode and not self.training: 
            mask = flatten_info
            this_num_rows = mask[0].sum()
            idx = torch.zeros(batch_size, max_row_num, dtype=int, device=mask.device)
            idx[:] = torch.arange(max_row_num, device=mask.device)
            batch_idx = torch.arange(mask.shape[0], device=mask.device)
            idx[batch_idx, mask.sum(1) - 1] = torch.arange(this_num_rows - 1, idx.shape[0] + 2, device=mask.device)
            idx = idx.reshape(-1)

            adjust_idx = torch.zeros(batch_size, max_row_num, device=mask.device, dtype=int)
            adjust_idx[batch_idx, mask.sum(1) - 1] = torch.arange(batch_size, device=mask.device)

            adjust_func = partial(self.flatten_adjust, idx1=adjust_idx.reshape(-1),
                                  idx2=torch.arange(max_row_num, device=mask.device).repeat(batch_size))

            args = tuple(a[idx] for a in args)
        return super().encode_unflatten(*args, batch_size=batch_size, max_row_num=max_row_num, sequence_len=sequence_len,
                                        flatten_info=flatten_info)[:-1] + (adjust_func,)

    def get_sd_pred(self, context_token_mask, context_encoding, schema_encoding,
                    table_encoding, table_mask, query_mask, is_first_iteration):
        use_header_detect = self.index_mode or \
            (self.union_mode and is_first_iteration and self.config.use_header_query_ffn_for_multi_union)
        if use_header_detect:
            sd_logits, e_id, r_id, c_id = self.header_query_detect(schema_encoding=schema_encoding,
                                                                   context_encoding=context_encoding,
                                                                   context_token_mask=context_token_mask,
                                                                   table_mask=table_mask,
                                                                   query_mask=query_mask)
            sd_pred = sd_logits > 0
            sd_pred = to_iob(sd_pred)
        else:
            logits, e_id, r_id, c_id = self.span_detect(final_table_encoding=table_encoding,
                                                        context_encoding=context_encoding,
                                                        context_token_mask=context_token_mask,
                                                        table_mask=table_mask,
                                                        query_mask=query_mask)
            sd_pred = logits.argmax(-1)
        return sd_pred, e_id, r_id, c_id

    def span_detect(self, context_encoding, final_table_encoding,
                    table_mask, context_token_mask, query_mask):
        e_id, r_id, c_id = torch.where(query_mask > -1)
        mask = table_mask[e_id, r_id, c_id].bool()  # in case MASK token got cut off
        e_id, r_id, c_id = e_id[mask], r_id[mask], c_id[mask]
        q_cell_enc = final_table_encoding[e_id, r_id, c_id]
        context_enc = context_encoding[e_id, r_id]
        context_mask = context_token_mask[e_id, r_id]

        B_logits = torch.matmul(self.sd_layer_B(q_cell_enc).unsqueeze(1), context_enc.permute(0, 2, 1)).squeeze(1)
        I_logits = torch.matmul(self.sd_layer_I(q_cell_enc).unsqueeze(1), context_enc.permute(0, 2, 1)).squeeze(1)
        O_logits = torch.zeros_like(B_logits, dtype=B_logits.dtype, device=B_logits.device) + self.sd_threshold

        logits = torch.stack((O_logits, I_logits, B_logits)).permute(1, 2, 0)
        logits[torch.where(1 - context_mask)] = torch.tensor([1000., -1000., -1000.], device=logits.device, dtype=logits.dtype)
        logits[:, 0] = torch.tensor([1000., -1000., -1000.], device=logits.device, dtype=logits.dtype)

        return logits, e_id, r_id, c_id


    def header_query_detect(self, context_encoding, schema_encoding,
                            table_mask, context_token_mask, query_mask):
        e_id, r_id, c_id = torch.where(query_mask > -1)
        mask = table_mask[e_id, r_id, c_id].bool()  # in case MASK token got cut off
        e_id, r_id, c_id = e_id[mask], r_id[mask], c_id[mask]
        q_col_enc = schema_encoding[e_id, c_id]
        context_enc = context_encoding[e_id, r_id]
        context_mask = context_token_mask[e_id, r_id]

        logits = torch.matmul(self.header_query_layer(q_col_enc).unsqueeze(1),
                              context_enc.permute(0, 2, 1)).squeeze(1) - self.header_query_threshold
        logits[torch.where(1 - context_mask)] = -1000
        logits[:, 0] = -1000

        return logits, e_id, r_id, c_id


    def get_span_embeddings(self, context_encoding, query_mask,
                            token_start_id, token_end_id, sample_id, row_id):
        span_lengths = (token_end_id - token_start_id)
        encodings = context_encoding[sample_id, row_id]
        span_mask = torch.zeros((encodings.size(0), encodings.size(1)), device=encodings.device)
        for i, (s, e) in enumerate(zip(token_start_id, token_end_id)):
            span_mask[i, s: e] = 1
            span_mask[i, s] += 1
            span_mask[i, e - 1] += 2
        
        embeddings =  self.span_embedding(context_encoding, sample_id, row_id, span_mask, span_lengths)
        return embeddings


class ELEETFinetuningModel(BaseEleetModel):

    def forward(self,
                column_token_position_to_column_ids,
                context_token_mask,
                context_token_positions,
                input_ids,
                sequence_mask,
                table_mask,
                query_mask,
                token_type_ids,
                query_labels,
                header_query_labels,
                deduplication_labels,
                **kwargs):
        context_encoding, schema_encoding, table_encoding, hidden_states = self.encode(
            column_token_position_to_column_ids=column_token_position_to_column_ids,
            context_token_mask=context_token_mask,
            context_token_positions=context_token_positions,
            input_ids=input_ids,
            sequence_mask=sequence_mask,
            table_mask=table_mask,
            token_type_ids=token_type_ids
        )

        deduplication_context_encoding = self.deduplication_encode(
            hidden_states[-2],
            column_token_position_to_column_ids=column_token_position_to_column_ids,
            context_token_mask=context_token_mask,
            context_token_positions=context_token_positions,
            input_ids=input_ids,
            sequence_mask=sequence_mask,
            table_mask=table_mask,
            token_type_ids=token_type_ids
        )

        sd_loss, sd_f1 = self.span_detect(
            final_table_encoding=table_encoding,
            context_encoding=context_encoding,
            context_token_mask=context_token_mask,
            table_mask=table_mask,
            query_mask=query_mask,
            query_labels=query_labels
        )

        hq_loss, hq_f1 = self.header_query_detect(
            schema_encoding=schema_encoding,
            context_encoding=context_encoding,
            context_token_mask=context_token_mask,
            table_mask=table_mask,
            query_mask=query_mask,
            header_query_labels=header_query_labels
        )

        dup_loss, dup_f1 = self.duplicate_detect(
            context_encoding=deduplication_context_encoding,
            deduplication_labels=deduplication_labels
        )


        mult_prefixes = ("sd", "hq", "dup")
        sd_mult, hq_mult, dup_mult = tuple(getattr(self, f"{x}_multiplier") for x in mult_prefixes)
        loss = sum(l * m for l, m in zip(
            (sd_loss, hq_loss, dup_loss),
            (sd_mult, hq_mult, dup_mult)
        ) if l is not None and not l.isnan())

        return ELEETFinetuningModelOutput(
            loss=loss,
            sd_loss_value=sd_loss,
            hq_loss_value=hq_loss,
            dup_loss_value=dup_loss,
            sd_f1=sd_f1,
            hq_f1=hq_f1,
            dup_f1=dup_f1
        )
    
    def span_detect(self, context_encoding, final_table_encoding,
                    table_mask, context_token_mask, query_mask, query_labels):
        e_id, r_id, c_id = torch.where(query_mask > -1)
        mask = table_mask[e_id, r_id, c_id].bool()  # in case MASK token got cut off
        e_id, r_id, c_id = e_id[mask], r_id[mask], c_id[mask]
        q_cell_enc = final_table_encoding[e_id, r_id, c_id]
        context_enc = context_encoding[e_id, r_id]
        context_mask = context_token_mask[e_id, r_id]

        B_logits = torch.matmul(self.sd_layer_B(q_cell_enc).unsqueeze(1), context_enc.permute(0, 2, 1)).squeeze(1)
        I_logits = torch.matmul(self.sd_layer_I(q_cell_enc).unsqueeze(1), context_enc.permute(0, 2, 1)).squeeze(1)
        O_logits = torch.zeros_like(B_logits, dtype=B_logits.dtype, device=B_logits.device) + self.sd_threshold

        logits = torch.stack((O_logits, I_logits, B_logits)).permute(1, 2, 0)
        logits[torch.where(1 - context_mask)] = torch.tensor([1000., -1000., -1000.], device=logits.device, dtype=logits.dtype)
        logits[:, 0] = torch.tensor([1000., -1000., -1000.], device=logits.device, dtype=logits.dtype)

        loss_fct = nn.CrossEntropyLoss()
        labels = query_labels[e_id, r_id, c_id].view(-1)
        loss = loss_fct(logits.view(-1, 3), labels)
        f1 = self.get_f1(pred=logits.view(-1, 3).argmax(-1), labels=labels)
        return loss, f1

    def header_query_detect(self, context_encoding, schema_encoding,
                            table_mask, context_token_mask, query_mask, header_query_labels):
        e_id, r_id, c_id = torch.where(query_mask > -1)
        mask = table_mask[e_id, r_id, c_id].bool()  # in case MASK token got cut off
        e_id, r_id, c_id = e_id[mask], r_id[mask], c_id[mask]
        q_col_enc = schema_encoding[e_id, c_id]
        context_enc = context_encoding[e_id, r_id]
        context_mask = context_token_mask[e_id, r_id]

        logits = torch.matmul(self.header_query_layer(q_col_enc).unsqueeze(1),
                              context_enc.permute(0, 2, 1)).squeeze(1) - self.header_query_threshold
        logits[torch.where(1 - context_mask)] = -1000
        logits[:, 0] = -1000

        loss_fct = nn.BCEWithLogitsLoss()
        labels = header_query_labels[e_id, r_id, c_id].clip(None, 1).float()
        loss = loss_fct(logits, labels)
        f1 = self.get_f1(pred=(logits > 0).int(), labels=labels)
        return loss, f1

    def duplicate_detect(self, context_encoding, deduplication_labels):
        deduplication_labels = deduplication_labels.reshape(-1, deduplication_labels.shape[-1])
        deduplication_labels = deduplication_labels[deduplication_labels[:, 0] > -1]
        labels = deduplication_labels[:, 0]
        batch_check = deduplication_labels[:, -1]

        # huggingface messes up the last few batches
        if len(batch_check) == 0 or batch_check[0] != len(context_encoding) or batch_check.min() != batch_check.max():
            return torch.tensor(0., device=context_encoding.device), \
                   torch.tensor(0., device=context_encoding.device)
        em_0, em_1 = self.get_span_embeddings(context_encoding, deduplication_labels)
        logits = compute_span_similarities(duplicate_detect_layer=self.duplicate_detect_layer,
                                           duplicate_detect_threshold=self.duplicate_detect_threshold,
                                           em_0=em_0, em_1=em_1)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        f1 = self.get_f1(pred=(logits > 0).int(), labels=labels)
        return loss, f1

    def get_span_embeddings(self, context_encoding, deduplication_labels):
        sample_id1 = deduplication_labels[:, 1]
        row_id1 = deduplication_labels[:, 2]
        token_start_id1 = deduplication_labels[:, 3]
        token_end_id1 = deduplication_labels[:, 4]
        sample_id2 = deduplication_labels[:, 5]
        row_id2 = deduplication_labels[:, 6]
        token_start_id2 = deduplication_labels[:, 7]
        token_end_id2 = deduplication_labels[:, 8]

        span_lengths1, span_mask1 = self.get_span_mask(context_encoding, sample_id1, row_id1, token_start_id1, token_end_id1)
        span_lengths2, span_mask2 = self.get_span_mask(context_encoding, sample_id2, row_id2, token_start_id2, token_end_id2)
        embeddings1 =  self.span_embedding(context_encoding, sample_id1, row_id1, span_mask1, span_lengths1)
        embeddings2 =  self.span_embedding(context_encoding, sample_id2, row_id2, span_mask2, span_lengths2)
        return embeddings1, embeddings2

    def get_span_mask(self, context_encoding, sample_id, row_id, token_start_id, token_end_id):
        span_lengths1 = (token_end_id - token_start_id)
        encodings = context_encoding[sample_id, row_id]
        span_mask1 = torch.zeros((encodings.size(0), encodings.size(1)), device=encodings.device)
        for i, (s, e) in enumerate(zip(token_start_id, token_end_id)):
            span_mask1[i, s: e] = 1
            span_mask1[i, s] += 1
            span_mask1[i, e - 1] += 2
        return span_lengths1, span_mask1

    def get_f1(self, pred, labels):
        tp = fp = fn = 0
        for label in labels.unique():
            if label == 0:
                continue
            tp += ((pred == label) & (labels == label)).sum()
            fp += ((pred == label) & (labels != label)).sum()
            fn += ((pred != label) & (labels == label)).sum()
        try:
            f1 = (2 * tp) / (2 * tp + fp + fn)
            return f1
        except ZeroDivisionError:
            return torch.tensor(0., device=pred.device)
