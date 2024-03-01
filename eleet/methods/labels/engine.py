from collections import defaultdict
import logging
from attr import field
from attrs import define
import numpy as np
import pandas as pd
from eleet.methods.base_engine import BaseEngine, EngineMode


logger = logging.getLogger(__name__)


@define
class LabelEngine(BaseEngine):
    max_result_tokens = 1024
    name = field(init=False, default="Labels")

    def setup(self):
        pass

    def shutdown(self):
        pass

    def max_prompt_length(self, x):
        return 0

    def execute(self, model_input, attributes, identifying_attribute, force_single_value_attributes, mode: EngineMode):
        data, normed, alignments, evidence_columns, report_column = model_input
        texts = data[report_column]
        span_labels = alignments[attributes].apply(
            lambda x: x.apply(
                lambda y: [texts.loc[[x.name[0] if isinstance(x.name, tuple) else x.name]].iloc[0][z[0]: z[1]].lower()
                           for z in y]), axis=1)
        normed = normed[attributes].applymap(lambda x: [y.lower() for y in x])
        result = span_labels.apply(lambda x: self.combine(x, normed.loc[x.name]), axis=1)
        if identifying_attribute is not None and identifying_attribute not in evidence_columns:  # more rows
            result.reset_index(result.index.names[1:], drop=True, inplace=True)
            index_map = {v: k for k, v in enumerate(model_input.data.index.unique(0))}
            result["index"] = [index_map[x] for x in result.index]
            result.set_index("index", inplace=True, drop=True)
        else:
            result = result.loc[data.index].reset_index(drop=True)
        return result

    def aggregate(self, col_values, normed):
        collect_normed = []
        collect_alternatives = []
        for _, (i, v) in col_values.iterrows():
            n = next(iter(y.lower() for x in normed.loc[[i]] for y in x if y.lower() in v))
            collect_normed.append(n)
            collect_alternatives.append(tuple(v))
        result = pd.DataFrame({"normed": collect_normed, "alternatives": collect_alternatives}) \
            .reset_index().groupby("normed").agg(tuple)
        result = {v["index"]: tuple({y for x in v["alternatives"] for y in x})
                  for _, v in result.iterrows()}
        return result
    
    def combine(self, row_span, row_normed):
        result_row = row_span.copy()
        for (span_idx, span_values), (normed_idx, normed_values) in zip(row_span.iteritems(), row_normed.iteritems()):
            assert span_idx == normed_idx
            result = defaultdict(set)
            for v_span, v_normed in zip(span_values, normed_values):
                result[v_normed].add(v_normed)
                result[v_normed].add(v_span)
            result_row[span_idx] = list(result.values())

        return result_row
