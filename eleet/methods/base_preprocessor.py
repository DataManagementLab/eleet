from contextlib import _GeneratorContextManager
from typing import List, Optional
from attrs import define
from eleet.database import Table
import pandas as pd


@define
class BasePreprocessor():
    finetuning_independent_of_operator = True

    def compute_model_input(self, data: Table, report_column: str, report_table_name: str, extract_attributes: List[str],
                            identifying_attribute: Optional[str], example_rows: Optional[pd.DataFrame],
                            multi_table: bool, limit:int = 2**32, index_mode: bool=False) -> _GeneratorContextManager:
        pass


    def get_evidence_columns(self, data, report_column):
        index_levels = data.data.index.levels if hasattr(data.data.index, "levels") else [data.data.index]
        evidence_columns = [l.name for l in index_levels if l.dtype != int] \
            + [c for c in data.data.columns if c != report_column]

        return evidence_columns
