from functools import partial
import torch
import logging
import inspect
from typing import Optional
from dataclasses import dataclass

from eleet_pretrain.model.base_model import BaseEleetModel
from transformers.file_utils import ModelOutput
from eleet_pretrain.model.table_decoding_dependencies import get_answer_end
from eleet_pretrain.utils import to_iob


MAX_NUM_SPAN_EMBEDDINGS = 50
logger = logging.getLogger(__name__)

@dataclass
class EleetModelOutput(ModelOutput):
    """Model output for ELEET transformers."""
    loss: Optional[torch.FloatTensor] = None
    idx: Optional[torch.IntTensor] = None
    table_id: Optional[torch.IntTensor] = None
    sub_idx: Optional[torch.IntTensor] = None
    sd_pred: Optional[torch.IntTensor] = None
    embedding: Optional[torch.FloatTensor] = None

class EleetModel(BaseEleetModel):

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

    def forward(self, **kwargs):
        if not self.training:
            return self.eval_forward(**kwargs)
        raise ValueError("Use BaseEleetModel for training")

    def eval_forward(self,  **kwargs):
        model_in = {k: v for k, v in kwargs.items() if k in set(inspect.getfullargspec(self.encode)[0][1:])}
        sd_in = {k: v for k, v in kwargs.items() if k in set(inspect.getfullargspec(self.span_detect)[0][1:])}
        stage = kwargs["stage"][0]

        context_encoding, schema_encoding, table_encoding, hidden_states = self.encode(**model_in)
        deduplication_context_encoding = self.deduplication_encode(hidden_states[-2], **model_in)
        sd_pred = self.get_sd_pred(stage, sd_in, context_encoding, schema_encoding, table_encoding)
        embeddings = self.get_span_embeddings(context_encoding=deduplication_context_encoding, sd_pred=sd_pred,
                                              **kwargs)

        return EleetModelOutput(loss=torch.tensor(0., device=sd_pred.device),
                               idx=kwargs["idx"],
                               table_id=kwargs["table_id"],
                               sub_idx=kwargs["sub_idx"],
                               sd_pred=sd_pred,
                               embedding=embeddings)

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

    def get_sd_pred(self, i, sd_in, context_encoding, schema_encoding, table_encoding):
        use_header_detect = self.index_mode or \
            self.union_mode and self.eval_stage == 0 and self.config.use_header_query_ffn_for_multi_union
        if use_header_detect:
            sd_logits, _ = self.header_query_detect(schema_encoding=schema_encoding,
                                                    context_encoding=context_encoding, **sd_in)
            sd_pred = sd_logits > 0
            sd_pred = to_iob(sd_pred)
        else:
            sd_logits, _ = self.span_detect(final_table_encoding=table_encoding,
                                            context_encoding=context_encoding, **sd_in)
            sd_pred = sd_logits.argmax(-1)
        return sd_pred

    def get_span_embeddings(self, context_encoding, query_mask, sd_pred, query_coords, **kw):
        b_id, q_num = torch.where(query_mask)
        num_interleave = (sd_pred[query_mask] == 2).sum(1)
        b_id = b_id.repeat_interleave(num_interleave)
        q_num = q_num.repeat_interleave(num_interleave)
        _, answer_start = torch.where(sd_pred[query_mask] == 2)
        r_id, _ = query_coords[b_id, q_num].T

        answer_end = get_answer_end(sd_pred, b_id, q_num, answer_start)
        embeddings = self._get_span_embeddings(context_encoding, b_id, r_id, answer_start, answer_end)

        result_embeddings = torch.full((context_encoding.shape[0], MAX_NUM_SPAN_EMBEDDINGS,
                                        embeddings.shape[-1]), -1000,
                                        dtype=embeddings.dtype, device=embeddings.device)
        for i in range(context_encoding.shape[0]):
            mask = (b_id == i)
            num_mask = mask.sum()
            if num_mask > MAX_NUM_SPAN_EMBEDDINGS:
                logger.warn("Too many span embeddings.")
            result_embeddings[i, :num_mask] = embeddings[mask][:MAX_NUM_SPAN_EMBEDDINGS]
        return result_embeddings

    def _get_span_embeddings(self, context_encoding, b_id, r_id, start, end):
        span_lengths = (end - start)
        encodings = context_encoding[b_id, r_id]
        span_mask = torch.zeros((encodings.size(0), encodings.size(1)), device=encodings.device)
        for i, (s, e) in enumerate(zip(start, end)):
            span_mask[i, s: e] = 1
            span_mask[i, s] += 1
            span_mask[i, e - 1] += 2
        
        return self.span_embedding(context_encoding, b_id, r_id, span_mask, span_lengths)