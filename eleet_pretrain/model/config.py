"""Configure multi-modal DB model."""

from copy import copy
import inspect
from itertools import chain
from transformers import AutoConfig
import ujson as json
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

from table_bert.config import TableBertConfig
from table_bert.vertical.config import VerticalAttentionTableBertConfig


class BaseEleetConfig():
    """Configuration class for ELEET model."""

    def _preinit(self, *args, max_len_evidence=120, max_len_answer=20, max_num_answers=20,
                 max_num_queries=50, max_num_cols=10, max_num_query_cols=7,
                 label_col=True, separate_evidence_from_text=False, 
                 disable_vertical_transform=False, window_overlap=100, cutout_col_label_prob=0.15,
                 disable_learned_deduplication=False, disable_header_query_ffn_for_multi_union=False,
                 max_num_deduplication_labels_per_row=20, deduplication_max_normed_len=10, **kwargs):
        """Initialize the configuration."""
        self.max_len_evidence = max_len_evidence
        self.max_len_answer = max_len_answer
        self.max_num_answers = max_num_answers
        # self.num_rnn_layers = num_rnn_layers
        # self.num_mlp_layers = num_mlp_layers
        # self.num_separate_layers = num_separate_layers
        # self.num_reinit = num_reinit
        # self.max_num_query_attrs = max_num_query_attrs
        self.max_num_query_cols = max_num_query_cols
        self.max_num_queries = max_num_queries
        self.max_num_cols = max_num_cols
        self.label_col = label_col
        self.separate_evidence_from_text = separate_evidence_from_text
        self.disable_vertical_transform = disable_vertical_transform
        self.window_overlap = window_overlap
        self.base_model_config = None
        self.cutout_col_label_prob = cutout_col_label_prob
        self.disable_learned_deduplication = disable_learned_deduplication
        self.disable_header_query_ffn_for_multi_union = disable_header_query_ffn_for_multi_union
        self.max_num_deduplication_labels_per_row = max_num_deduplication_labels_per_row
        self.deduplication_max_normed_len = deduplication_max_normed_len
        return args, kwargs

    @property
    def enable_learned_deduplication(self):
        return not self.disable_learned_deduplication

    @property
    def use_header_query_ffn_for_multi_union(self):
        return not self.disable_header_query_ffn_for_multi_union

    @property
    def total_num_cols(self):
        return self.max_num_cols + self.label_col
    
    def _post_init(self):
        if self.separate_evidence_from_text:
            self.max_len_evidence = self.max_sequence_len

    def set_base_model_config(self, base_model_config):
        self.base_model_config = base_model_config

    def get_base_model_config(self):
        return self.base_model_config or AutoConfig.from_pretrained(self.base_model_name)

    def get_default_values_for_parameters(self):
        signature1 = inspect.signature(self.__init__)
        signature2 = inspect.signature(self._preinit)

        default_args = OrderedDict(
            (k, v.default)
            for k, v in chain(signature1.parameters.items(), signature2.parameters.items())
            if v.default is not inspect.Parameter.empty
        )

        return default_args

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """Add arguments to argument parser."""
        parser.add_argument('--max-len-text', default=400, type=int)
        parser.add_argument('--max-len-answers', default=20, type=int)
        parser.add_argument('--max-num-answers', default=10, type=int)
        parser.add_argument('--num-rnn-layers', default=2, type=int)
        parser.add_argument('--num-separate_layers', default=2, type=int)
        parser.add_argument('--num-reinit', default=2, type=int)
        parser.add_argument('--max-num-query-attributes', default=3, type=int)
        return parser

    def to_json_string(self):
        result = copy(vars(self))
        del result["base_model_config"]
        return json.dumps(result)

    def save_pretrained(self, save_directory):
        self.save(Path(save_directory) / 'tb_config.json')


class AutoEleetConfig():

    @staticmethod
    def from_file(config_file):
        if VerticalEleetConfig.is_valid_config_file(config_file):
            return VerticalEleetConfig.from_file(config_file)
        # return BaseEleetConfig.from_file(config_file)
        

# class BaseEleetConfig(BaseEleetConfig, TableBertConfig):
#     def __init__(self, *args, **kwargs):
#         if "predict_cell_tokens" in kwargs:
#             del kwargs["predict_cell_tokens"]
#         args, kwargs = BaseEleetConfig._preinit(self, predict_cell_tokens=True, *args, **kwargs)
#         TableBertConfig.__init__(self, *args, **kwargs)
#         BaseEleetConfig._post_init(self)


class VerticalEleetConfig(BaseEleetConfig, VerticalAttentionTableBertConfig):
    def __init__(self, *args, **kwargs):
        if "predict_cell_tokens" in kwargs:
            del kwargs["predict_cell_tokens"]
        args, kwargs = BaseEleetConfig._preinit(self, *args, **kwargs)
        VerticalAttentionTableBertConfig.__init__(self, predict_cell_tokens=True, *args, **kwargs)
        BaseEleetConfig._post_init(self)
