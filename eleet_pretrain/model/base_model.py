"""Transformer model for joining tabular and text data."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from collections import OrderedDict

import numpy as np
from table_bert.vanilla_table_bert import VanillaTableBert
from table_bert.vertical.config import VerticalAttentionTableBertConfig
import torch
from torch import nn
from transformers import (AutoConfig, AutoModelForMaskedLM, BertForMaskedLM, PreTrainedModel)
from transformers.file_utils import ModelOutput
from eleet_pretrain.model.config import AutoEleetConfig, BaseEleetConfig, VerticalEleetConfig
from eleet_pretrain.utils import debug_transform, debug_transform_binary, get_no_answers, to_iob, visualize_single, DebugUnderlining
from table_bert.vertical.vertical_attention_table_bert import BertVerticalLayer, VerticalAttentionTableBert, VerticalEmbeddingLayer
from torch_scatter import scatter_mean
from copy import deepcopy


logger = logging.getLogger(__name__)


@dataclass
class BaseEleetModelOutput(ModelOutput):
    """Model output for ELEET transformers."""
    loss: Optional[torch.FloatTensor] = None
    sd_loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    hq_loss: Optional[torch.FloatTensor] = None
    dup_loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    sd_out: Optional[torch.FloatTensor] = None
    mlm_out: Optional[torch.FloatTensor] = None
    # schema_out: Optional[torch.FloatTensor] = None
    rt_loss: Optional[torch.FloatTensor] = None
    idx: Optional[torch.IntTensor] = None
    table_id: Optional[torch.IntTensor] = None
    text_ids: Optional[torch.IntTensor] = None
    span_embeddings: Optional[torch.IntTensor] = None


class BaseEleetModel(VerticalAttentionTableBert, PreTrainedModel):
    """Base model."""
    def __init__(self,
                 config: VerticalAttentionTableBertConfig,
                 **kwargs):
        super(VanillaTableBert, self).__init__(config, **kwargs)

        self._bert_model = BertForMaskedLM(config.get_base_model_config())
        self.vertical_layers_enabled = not config.disable_vertical_transform

        if self.vertical_layers_enabled:
            self.vertical_embedding_layer = VerticalEmbeddingLayer()
            self.vertical_transformer_layers = nn.ModuleList([
                BertVerticalLayer(self.config)
                for _ in range(self.config.num_vertical_layers)
            ])

        self.sd_layer_B = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sd_layer_I = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sd_threshold = nn.Parameter(torch.tensor([0.]))
        self.relevant_text_detect_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.relevant_text_threshold = nn.Parameter(torch.tensor([0.]))
        self.header_query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.header_query_threshold = nn.Parameter(torch.tensor([0.]))

        self.duplicate_detect_bert_layer = deepcopy(self.bert.encoder.layer[-1])
        self.headword_detector = nn.Linear(config.hidden_size, 1)
        self.span_compress_layer = nn.Linear(config.hidden_size * 4, int(config.hidden_size))
        self.duplicate_detect_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.duplicate_detect_threshold = nn.Parameter(torch.tensor([0.]))
        self.span_detect_num_labels = 3

        if self.vertical_layers_enabled:
            added_modules = [self.vertical_embedding_layer, self.vertical_transformer_layers]
            for module in added_modules:
                    module.apply(self._bert_model._init_weights)

        self.config.keys_to_ignore_at_inference = ["sd_loss", "mlm_loss", "hq_loss", "rt_loss", "dup_loss", "cls_loss"]

        self.rng = np.random.default_rng(42)
        self.debug_fraction = 0
        self.sd_multiplier = kwargs.get("sd_loss_multiplier", 1.) 
        self.mlm_multiplier = kwargs.get("mlm_loss_multiplier", 1.)
        self.hq_multiplier = kwargs.get("hq_loss_multiplier", 1.)
        self.rt_multiplier = kwargs.get("rt_loss_multiplier", 1.)
        self.dup_multiplier = kwargs.get("dup_loss_multiplier", 1.)
        self.cls_multiplier = kwargs.get("cls_loss_multiplier", 1.)

    def set_debug_fraction(self, debug_fraction):
        self.debug_fraction = debug_fraction

    def freeze_layers(self, num):
        all_layers = self._bert_model.bert.encoder.layer
        logger.info(f"Freezing the {num} first layers of the{len(all_layers)} layers!")
        for i in range(num):
            for param in self._bert_model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self,
                idx, table_id,
                query_labels,
                query_coords,
                column_token_position_to_column_ids,
                context_token_mask,
                context_token_positions,
                input_ids,
                sequence_mask,
                table_mask,
                token_type_ids,
                text_ids=None,
                normalized_answers=None,
                masked_context_token_labels=None,
                header_query_labels=None,
                header_query_coords=None,
                relevant_text_labels=None,
                deduplication=None,
                **kwargs):
        """Forward pass."""

        context_encoding, schema_encoding, final_table_encoding, hidden_states = self.encode(
            column_token_position_to_column_ids=column_token_position_to_column_ids,
            context_token_mask=context_token_mask,
            context_token_positions=context_token_positions,
            input_ids=input_ids,
            sequence_mask=sequence_mask,
            table_mask=table_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        deduplication_context_encoding = None
        if deduplication is not None:
            deduplication_context_encoding = self.deduplication_encode(hidden_states[-2], context_token_mask,
                                                                       context_token_positions, sequence_mask)

        query_mask = (query_coords[:, :, 0] >= 0) & (query_coords[:, :, 3] == -1)
        sd_logits, sd_loss = self.span_detect(
            context_encoding=context_encoding,
            final_table_encoding=final_table_encoding,
            query_labels=query_labels,
            query_coords=query_coords[:, :, :2],
            table_mask=table_mask,
            query_mask=query_mask,
            context_token_mask=context_token_mask
        )

        header_query_mask = None
        if header_query_coords is not None:
            header_query_mask = (header_query_coords[:, :, 0] >= 0)

        mlm_logits, mlm_loss, hq_logits, hq_loss, rt_logits, rt_loss, dup_logits, dup_loss, cls_loss = \
            self.secondary_objectives(
                header_query_labels=header_query_labels,
                header_query_coords=header_query_coords,
                header_query_mask=header_query_mask,
                relevant_text_labels=relevant_text_labels,
                context_token_mask=context_token_mask,
                table_mask=table_mask,
                masked_context_token_labels=masked_context_token_labels,
                context_encoding=context_encoding,
                deduplication_context_encoding=deduplication_context_encoding,
                schema_encoding=schema_encoding,
                final_table_encoding=final_table_encoding,
                deduplication_labels=deduplication
            )

        if logging.root.level <= logging.DEBUG and self.rng.random() < self.debug_fraction:
            self.debug_print(input_ids=input_ids, token_type_ids=token_type_ids, sequence_mask=sequence_mask,
                             normalized_answers=normalized_answers, query_labels=query_labels,
                             query_coords=query_coords, is_training=self.training,
                             masked_context_token_labels=masked_context_token_labels,
                             header_query_labels=header_query_labels, header_query_coords=header_query_coords,
                             relevant_text_labels=relevant_text_labels, mlm_logits=mlm_logits,
                             sd_logits=sd_logits, rt_logits=rt_logits, hq_logits=hq_logits, dup_labels=deduplication,
                             dup_logits=dup_logits)

        mult_prefixes = ("sd", "mlm", "hq", "rt", "dup", "cls")
        sd_mult, mlm_mult, hq_mult, rt_mult, dup_mult, cls_mult = tuple(getattr(self, f"{x}_multiplier")
                                                                        for x in mult_prefixes)
        loss = sum(l * m for l, m in zip(
            (sd_loss, mlm_loss, hq_loss, rt_loss, dup_loss, cls_loss),
            (sd_mult, mlm_mult, hq_mult, rt_mult, dup_mult, cls_mult)
        ) if l is not None and not l.isnan())

        return BaseEleetModelOutput(
            loss=loss,
            sd_loss=sd_loss * sd_mult if sd_loss is not None else None,
            mlm_loss=mlm_loss * mlm_mult if mlm_loss is not None else None,
            hq_loss=hq_loss * hq_mult if hq_loss is not None else None,
            rt_loss=rt_loss * rt_mult if rt_loss is not None else None,
            dup_loss=dup_loss * dup_mult if dup_loss is not None else None,
            cls_loss=cls_loss * cls_mult if cls_loss is not None else None,
            sd_out=sd_logits.argmax(-1),
            mlm_out=mlm_logits.argmax(-1) if mlm_logits is not None else None,
            idx=idx, table_id=table_id, text_ids=text_ids
        )

    def secondary_objectives(self,
                             header_query_labels,
                             header_query_coords,
                             header_query_mask,
                             relevant_text_labels,
                             context_token_mask,
                             table_mask,
                             masked_context_token_labels,
                             context_encoding,
                             deduplication_context_encoding,
                             schema_encoding,
                             final_table_encoding,
                             deduplication_labels):

        mlm_logits, mlm_loss, hq_logits, hq_loss, rt_logits, rt_loss, dup_logits, dup_loss, cls_loss = (None,) * 9
        if masked_context_token_labels is not None:
            mlm_logits, mlm_loss = self.mlm(masked_context_token_labels, context_encoding)
        if header_query_labels is not None:
            hq_logits, hq_loss = self.header_query_detect(
                context_encoding=context_encoding,
                schema_encoding=schema_encoding,
                query_labels=header_query_labels,
                query_coords=header_query_coords,
                table_mask=table_mask, 
                query_mask=header_query_mask,
                context_token_mask=context_token_mask
            )
        if relevant_text_labels is not None:
            rt_logits, rt_loss = self.relevant_text_detect(context_encoding, context_token_mask,
                                                           final_table_encoding, relevant_text_labels)
        if deduplication_labels is not None:
            dup_logits, dup_loss = self.duplication_detect(
                deduplication_context_encoding=deduplication_context_encoding,
                deduplication_labels=deduplication_labels
            )
                                                           
        return mlm_logits, mlm_loss, hq_logits, hq_loss, rt_logits, rt_loss, dup_logits, dup_loss, cls_loss

    def mlm(self, masked_context_token_labels, context_encoding):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        context_token_scores = self._bert_model.cls.predictions(context_encoding)
        masked_context_token_loss = loss_fct(context_token_scores.view(-1, self.config.vocab_size),
                                             masked_context_token_labels.view(-1))
        return context_token_scores, masked_context_token_loss

    def relevant_text_detect(self, context_encoding, context_token_mask, table_encoding, labels):
        row_encoding = (table_encoding.sum(-2) / table_encoding.size(-2)).unsqueeze(2)
        logits = torch.matmul(self.relevant_text_detect_layer(row_encoding),
                              context_encoding.permute(0, 1, 3, 2)).squeeze(2) - self.relevant_text_threshold
        logits[torch.where(1 - context_token_mask)] = -1000
        logits[:, :, 0] = -1000
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return logits, loss

    def duplication_detect(self, deduplication_context_encoding, deduplication_labels):
        the_sum = (deduplication_labels > 0).int().sum(-1)
        b_id, r_id, cls_label, d_id = torch.where(the_sum.all(-1) > 0)
        dup_labels = deduplication_labels[b_id, r_id, cls_label, d_id]

        span_lengths = the_sum[b_id, r_id, cls_label, d_id]
        em_0 = self.span_embedding(deduplication_context_encoding, b_id, r_id, dup_labels[:, 0], span_lengths[:, 0])
        em_1 = self.span_embedding(deduplication_context_encoding, b_id, r_id, dup_labels[:, 1], span_lengths[:, 0])
        em_0 = self.duplicate_detect_layer(em_0)

        logits = (em_0 * em_1).sum(1) # / (torch.norm(em_0, p=2, dim=1) * torch.norm(em_1, p=2, dim=1))
        logits = logits - self.duplicate_detect_threshold
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, 1. - cls_label.float())
        return logits, loss if not torch.isnan(loss) else torch.tensor(0., device=loss.device)

    def span_embedding(self, context_encoding, b_id, r_id, dup_labels, span_lengths):
        encodings = context_encoding[b_id, r_id]
        # headwords
        headwords = self.headword_detector(context_encoding).squeeze(-1)[b_id, r_id]
        weights = nn.Softmax(dim=1)(headwords - 1000 * (1 - (dup_labels > 0).int()))
        head_emb = (encodings * weights.unsqueeze(-1)).sum(1)

        # boundaries
        idx = torch.arange(dup_labels.size(0), device=dup_labels.device)
        _1, start_span = torch.where((dup_labels == 2) | (dup_labels == 4))
        _2, end_span = torch.where((dup_labels == 3) | (dup_labels == 4))
        assert (_1 == idx).all() and (_2 == idx).all() 
        start_emb = encodings[idx, start_span]
        end_emb = encodings[idx, end_span]

        # span lengths
        frequency_div = (10_000 ** (
            2 * torch.arange(int(self.config.hidden_size / 2), device=span_lengths.device) / self.config.hidden_size))
        sin_emb = torch.sin(span_lengths.reshape(-1, 1) / frequency_div)
        cos_emb = torch.cos(span_lengths.reshape(-1, 1) / frequency_div)

        em = self.span_compress_layer(torch.hstack((start_emb, end_emb, head_emb, sin_emb, cos_emb)))
        return em

    def header_query_detect(self, context_encoding, schema_encoding, query_labels, query_coords, table_mask, query_mask,
                            context_token_mask):
        e_id, q_id = torch.where(query_mask)
        r_id, c_id = query_coords[e_id, q_id].T
        mask = table_mask[e_id, r_id, c_id].bool()  # in case MASK token got cut off
        e_id, q_id, r_id, c_id = e_id[mask], q_id[mask], r_id[mask], c_id[mask]
        q_col_enc = schema_encoding[e_id, c_id]
        context_enc = context_encoding[e_id, r_id]
        context_mask = context_token_mask[e_id, r_id]

        logits = torch.matmul(self.header_query_layer(q_col_enc).unsqueeze(1),
                              context_enc.permute(0, 2, 1)).squeeze(1) - self.header_query_threshold
        logits[torch.where(1 - context_mask)] = -1000
        logits[:, 0] = -1000
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, (query_labels[e_id, q_id] > 0).float()) if query_labels != None else None

        result = torch.zeros(len(context_encoding), self.config.max_num_queries, self.config.max_sequence_len,
                             device=logits.device, dtype=logits.dtype)
        result[e_id, q_id] = logits
        return result, loss

    def span_detect(self, context_encoding, final_table_encoding, query_labels, query_coords, table_mask, query_mask,
                    context_token_mask):
        e_id, q_id = torch.where(query_mask)
        r_id, c_id = query_coords[e_id, q_id].T
        mask = table_mask[e_id, r_id, c_id].bool()  # in case MASK token got cut off
        e_id, q_id, r_id, c_id = e_id[mask], q_id[mask], r_id[mask], c_id[mask]
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
        loss = loss_fct(logits.view(-1, 3), query_labels[e_id, q_id].view(-1))

        result = torch.zeros(len(context_encoding), self.config.max_num_queries, self.config.max_sequence_len, 3,
                             device=logits.device, dtype=logits.dtype)
        result[e_id, q_id] = logits
        return result, loss

    def encode_flatten(self, *args, batch_size, max_row_num, mask):
        return tuple(a.view(batch_size * max_row_num, -1) for a in args) + (None,)

    def encode_unflatten(self, *args, batch_size, max_row_num, sequence_len, flatten_info):
        return tuple(a.view(batch_size, max_row_num, sequence_len, -1) for a in args) + (None,)

    def encode(self,
               column_token_position_to_column_ids,
               context_token_mask,
               context_token_positions,
               input_ids,
               sequence_mask,
               table_mask,
               token_type_ids,
               **kwargs):
        """Forward pass."""
        batch_size, max_row_num, sequence_len = input_ids.size()

        mask = sequence_mask[:, :, 0] > 0
        flattened_input_ids, flattened_token_type_ids, flattened_sequence_mask, flatten_info = self.encode_flatten(
            input_ids, token_type_ids, sequence_mask, max_row_num=max_row_num, batch_size=batch_size, mask=mask)

        bert_output, hidden_states = self.bert(
            input_ids=flattened_input_ids,
            token_type_ids=flattened_token_type_ids,
            attention_mask=flattened_sequence_mask,
            output_hidden_states=True,
            **kwargs
        )[0:2]

        # (batch_size, max_row_num, sequence_len, hidden_size)
        bert_output, adjust_func = self.encode_unflatten(bert_output, batch_size=batch_size, max_row_num=max_row_num,
                                                         sequence_len=sequence_len, flatten_info=flatten_info)
        if adjust_func is not None:
            column_token_position_to_column_ids = adjust_func(column_token_position_to_column_ids)
            context_token_mask = adjust_func(context_token_mask)
            context_token_positions = adjust_func(context_token_positions)
            input_ids = adjust_func(input_ids)
            sequence_mask = adjust_func(sequence_mask)
            table_mask = adjust_func(table_mask)
            token_type_ids = adjust_func(token_type_ids)

        # expand to the same size as `bert_output`
        column_token_to_column_id_expanded = column_token_position_to_column_ids.unsqueeze(-1).expand(
            -1, -1, -1, bert_output.size(-1)  # (batch_size, max_row_num, sequence_len, hidden_size)
        ).contiguous()

        # (batch_size, max_row_num, max_column_num, hidden_size)
        max_column_num = table_mask.size(-1)
        column_token_to_column_id_expanded[column_token_to_column_id_expanded == -1] = max_column_num
        table_encoding = scatter_mean(
            src=bert_output,
            index=column_token_to_column_id_expanded,
            dim=-2,  # over `sequence_len`
            dim_size=max_column_num + 1   # last dimension is the used for collecting unused entries
        )
        table_encoding = table_encoding[:, :, :-1, :] * table_mask.unsqueeze(-1)

        context_encoding = torch.gather(
            bert_output,
            dim=-2,
            index=context_token_positions.unsqueeze(-1).expand(-1, -1, -1, bert_output.size(-1)),
        )

        # expand to (batch_size, max_row_num, max_context_len)
        # context_token_mask = context_token_mask.unsqueeze(1).expand(-1, max_row_num, -1)
        context_encoding = context_encoding * context_token_mask.unsqueeze(-1)

        if self.vertical_layers_enabled:              # perform vertical attention
            table_encoding = self.vertical_transform_only_table(table_encoding, table_mask)

        # mean-pool last encodings (batch_size, 1, 1)
        table_row_nums = table_mask[:, :, 0].sum(dim=-1)[:, None, None]
        schema_encoding = table_encoding.sum(dim=1) / table_row_nums  # (batch_size, max_column_num, hidden_size)

        return context_encoding, schema_encoding, table_encoding, hidden_states

    def deduplication_encode(self, hidden_state,
                             context_token_mask,
                             context_token_positions,
                             sequence_mask,
                             **kwargs):
        batch_size, max_row_num, sequence_len = context_token_mask.size()

        mask = sequence_mask[:, :, 0] > 0
        flattened_sequence_mask, flatten_info = self.encode_flatten(sequence_mask, max_row_num=max_row_num,
                                                                    batch_size=batch_size, mask=mask)

        deduplication_bert_output = self.duplicate_detect_bert_layer(
            hidden_state,
            attention_mask=self.bert.get_extended_attention_mask(flattened_sequence_mask,
                                                                 flattened_sequence_mask.size())
        )[0]

        deduplication_bert_output, adjust_func = self.encode_unflatten(
            deduplication_bert_output, batch_size=batch_size, max_row_num=max_row_num, sequence_len=sequence_len,
            flatten_info=flatten_info)

        if adjust_func is not None:
            context_token_mask = adjust_func(context_token_mask)
            context_token_positions = adjust_func(context_token_positions)
            sequence_mask = adjust_func(sequence_mask)

        deduplication_context_encoding = torch.gather(
            deduplication_bert_output,
            dim=-2,
            index=context_token_positions.unsqueeze(-1).expand(-1, -1, -1, deduplication_bert_output.size(-1)),
        )

        deduplication_context_encoding = deduplication_context_encoding * context_token_mask.unsqueeze(-1)
        return deduplication_bert_output

    def vertical_transform_only_table(self, table_encoding, table_mask):
        # (batch_size, max_row_num, sequence_len)
        sequence_mask = table_mask

        # (batch_size, sequence_len, 1, max_row_num, 1)
        attention_mask = sequence_mask.permute(0, 2, 1)[:, :, None, :, None]
        attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_state = table_encoding
        for vertical_layer in self.vertical_transformer_layers:
            hidden_state = vertical_layer(hidden_state, attention_mask=attention_mask)

        last_hidden_state = hidden_state * sequence_mask.unsqueeze(-1)

        return last_hidden_state


    @staticmethod
    def get_configs(  # pylint: disable=arguments-differ
        model_name_or_path: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        config: Optional[BaseEleetConfig] = None,
    ):
        """Load pre-trained weights and config."""

        # Load the TaBERT config
        if not isinstance(config, BaseEleetConfig):  # Copied from table_bert.py
            if config_file:
                config_file = Path(config_file)
            else:
                assert model_name_or_path, f'model path is None'  # noqa
                config_file = Path(model_name_or_path).parent / 'tb_config.json'

            if config_file.exists():
                config = AutoEleetConfig.from_file(config_file)
            else:
                logger.warn(f"Could not find config file {config_file}. "
                            f"Using default TaBERT config for base model {model_name_or_path}.")
                config = VerticalEleetConfig()
                config.base_model_name = model_name_or_path

        # Load the model's pre-trained weights. Either from disk or from transformers library.
        base_model_config = AutoConfig.from_pretrained(config.base_model_name)
        config.set_base_model_config(base_model_config)
        return config
    
    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: Optional[Union[str, Path]] = None,
                        config_file: Optional[Union[str, Path]] = None,
                        config: Optional[BaseEleetConfig] = None,
                        strict: bool = False,
                        **kwargs):
        """Load from TaBERT pretrained weights"""
        tb_config = BaseEleetModel.get_configs(model_name_or_path, config_file, config)
        if Path(model_name_or_path).exists():
            state_dict = torch.load(model_name_or_path, map_location="cpu")
        else:
            m = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
            state_dict = OrderedDict((f"_bert_model.{k}", v) for k,v in m.state_dict().items())
            model_name_or_path = ""
        # pooler not instantiated in latest transformer version
        # not used for TaBERT anyways
        if "_bert_model.bert.pooler.dense.weight" in state_dict:
            del state_dict["_bert_model.bert.pooler.dense.weight"]
            del state_dict["_bert_model.bert.pooler.dense.bias"]
            # position id's are exported in latest transformer version. Is not trained, so can be set to starting position.
            state_dict["_bert_model.bert.embeddings.position_ids"] = BertForMaskedLM(
                AutoConfig.from_pretrained("bert-base-uncased")).bert.embeddings.position_ids
            # prediction.decoder.bias is set to prediction.bias in latest transformer version
            # span based prediction not used
            for k in list(state_dict):
                if k.startswith("span_based_prediction"):
                    del state_dict[k]
        model = super(BaseEleetModel, cls).from_pretrained(
            model_name_or_path=model_name_or_path,
            state_dict=state_dict,
            config=tb_config,
            strict=strict,
            **kwargs
        )
        if not any("duplicate_detect_bert_layer" in k for k in state_dict.keys()):
            model.duplicate_detect_bert_layer.load_state_dict(model.bert.encoder.layer[-1].state_dict())
        return model

    def get_trainer(self, *args, **kwargs):
        from eleet_pretrain.model.trainer import EleetTrainer
        return EleetTrainer(*args, **kwargs)

    def debug_print(self, input_ids, token_type_ids, sequence_mask, normalized_answers, query_labels,
                    query_coords, is_training, masked_context_token_labels, header_query_labels,
                    header_query_coords, relevant_text_labels, mlm_logits, sd_logits,
                    rt_logits, hq_logits, dup_labels, dup_logits):
        device_logger = self.setup_logger(input_ids.device)
        for i in range(input_ids.size(0)):
            underlinings = list()
            answer_start, answer_end, answer_col_ids = debug_transform(input_ids[i], query_labels[i], query_coords[i])
            no_answers = get_no_answers(query_labels[i], query_coords[i])
            pred_start, pred_end, pred_col_ids = debug_transform(input_ids[i], sd_logits[i].argmax(-1), query_coords[i])
            underlinings += [
                DebugUnderlining("Masked Cells", "M", answer_start, answer_end, answer_col_ids,
                                 normalized_answers[i] if normalized_answers is not None else None,
                                 no_answers=no_answers),
                DebugUnderlining("Prediction", "P", pred_start, pred_end, pred_col_ids),
            ]

            if relevant_text_labels is not None:
                rt_label_start, rt_label_end, _ = debug_transform_binary(input_ids[i], relevant_text_labels[i], None)
                rt_pred_start, rt_pred_end, _ = debug_transform_binary(input_ids[i], rt_logits[i] > 0, None)
                underlinings += [
                    DebugUnderlining("Relevant Text", "R", rt_label_start, rt_label_end, None),
                    DebugUnderlining("Predicted Relevance", "PR", rt_pred_start, rt_pred_end, None),
                ]

            if header_query_labels is not None:
                hq_start, hq_end, hq_col_ids = debug_transform(input_ids[i], header_query_labels[i], header_query_coords[i])
                no_answers = get_no_answers(header_query_labels[i], header_query_coords[i])
                hq_pred_start, hq_pred_end, hq_pred_col_ids = debug_transform_binary(input_ids[i], hq_logits[i] > 0, header_query_coords[i])
                underlinings += [
                    DebugUnderlining("Aligned Header", "H", hq_start, hq_end, hq_col_ids, no_answers=no_answers),
                    DebugUnderlining("Predicted Header Alignment", "PH", hq_pred_start, hq_pred_end, hq_pred_col_ids)
                ]

            mlm_pred = None
            if mlm_logits is not None:
                mlm_pred = mlm_logits[i].argmax(-1)
                mlm_pred[masked_context_token_labels[i] == -1] = -1
            mlm_labels = None
            if masked_context_token_labels is not None:
                mlm_labels = masked_context_token_labels[i]

            visualize_single(
                self.tokenizer,
                input_ids=input_ids[i],
                token_type_ids=token_type_ids[i],
                sequence_mask=sequence_mask[i],
                is_training=is_training,
                masked_context_token_labels=mlm_labels,
                pred_mlm=mlm_pred,
                print_func=device_logger.debug,
                underlinings=underlinings
            )
        if dup_labels is not None:
            self.debug_print_deduplication(device_logger, input_ids, dup_labels, dup_logits)

    def debug_print_deduplication(self, device_logger, input_ids, dup_labels, dup_logits):
        device_logger.debug("Deduplication:")
        for b, r, t, n, pred in zip(*torch.where(dup_labels[:, :, :, :].any(-1).all(-1)), dup_logits):
            device_logger.debug(" ".join((
                f"B{b} R{r}",
                self.tokenizer.decode(input_ids[b, r][dup_labels[b, r, t, n, 0].bool()]),
                "!=" if t else "==",
                self.tokenizer.decode(input_ids[b, r][dup_labels[b, r, t, n, 1].bool()]),
                "| Model prediction:",
                "==" if pred > 0 else "!="
            )))

    def setup_logger(self, device):
        device_logger = logging.getLogger(f"{__file__}-{device}")
        if len(device_logger.handlers) == 0 and len(logging.root.handlers) > 0:
            fn = Path(logging.root.handlers[0].baseFilename)
            fh = logging.FileHandler(fn.parent / f"{device}-{fn.name}")
            device_logger.addHandler(fh)
        device_logger.propagate = False
        return device_logger
