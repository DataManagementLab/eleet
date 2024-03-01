from collections import namedtuple
from contextlib import contextmanager
import hashlib
import logging
import os

from attr import field
import numpy as np
import pandas as pd
from eleet.methods.base_preprocessor import BasePreprocessor
from attrs import define


logger = logging.getLogger(__name__)


ModelInput = namedtuple("ModelInput", ["data", "normed", "alignments", "evidence_columns", "report_column"])

@define
class LabelPreprocessor(BasePreprocessor):

    @contextmanager
    def compute_model_input(self, data, *args, report_column, limit=2**32, **kwargs):
        evidence_columns = self.get_evidence_columns(data, report_column)
        index = data.data.index[:limit] if not isinstance(data.data.index, pd.MultiIndex) else \
            data.data.index.unique(0)[:limit]
        labels_index = [i for i in index if i in data.labels.normed.index]
        yield ModelInput(data=data.data.loc[index],
                         normed=data.labels.normed.loc[labels_index],
                         alignments=data.labels.alignments.loc[labels_index],
                         evidence_columns=evidence_columns,
                         report_column=report_column)
