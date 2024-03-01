from abc import ABC, abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
import enum
from pathlib import Path
import pickle
from typing import Dict

from attr import define, field
import pandas as pd


EngineModeTuple = namedtuple("EngineModeTuple", ["union_mode", "fast_mode", "index_mode"])
class EngineMode(enum.Enum):
    JOIN = EngineModeTuple(False, False, False)
    UNION = EngineModeTuple(True, True, False)
    SCAN = EngineModeTuple(True, False, False)
    INDEX = EngineModeTuple(False, False, True)


@define
class BaseEngine(ABC):
    index_build_time: timedelta = field(init=False, default=timedelta())
    runtime_correction: timedelta = field(init=False, default=timedelta())
    cache_dir: Path = field(converter=Path)
    cache: Dict = field(init=False, factory=dict)
    cache_file = field(init=False, default=None)
    translate_start_time = field(init=False, default=0)
    use_normed_conditions = True

    def setup(self):
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_file = self.get_cache_file()
        if self.cache_file.exists():
            with self.cache_file.open("rb") as f:
                self.cache = pickle.load(f)

    def shutdown(self):
        with self.cache_file.open("wb") as f:
            pickle.dump(self.cache, f)

    def check_cache(self, key):
        cached = self.cache.get(key)
        if cached:
            result, runtime = cached
            self.runtime_correction += runtime
            print("use cached")
            return result
        self.translate_start_time = datetime.now()

    def update_cache(self, key, do_update, result):
        if not do_update:
            return
        runtime = datetime.now() - self.translate_start_time
        self.cache[key] = (result, runtime)

    @abstractmethod
    def execute(self, model_input, attributes, identifying_attribute, force_single_value_attributes, table_name, mode):
        """Execute extraction from texts."""

    def get_empty_result(self, data, extract_attributes):
        result = pd.DataFrame(columns=extract_attributes)
        result.index.name = (data.index.name)
        return result

    def aggregate(self, col_values):
        result = {
            v: k
            for k, v in col_values.reset_index(drop=True).reset_index().groupby(col_values.name).agg(tuple)["index"].iteritems()
        }
        return result
