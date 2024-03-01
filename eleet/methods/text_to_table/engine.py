from collections import defaultdict
import hashlib
import os
from pathlib import Path
import re
import subprocess
from attr import define, field
from fuzzywuzzy import fuzz
import pandas as pd
import torch
from eleet.methods.base_engine import BaseEngine, EngineMode

NEW_LINE = " <NEWLINE> "
SEPARATOR_STRING = '|'


@define
class T2TEngine(BaseEngine):
    model_path = field(converter=lambda x: Path(x).absolute())
    name = field(init=False, default="Text-To-Table")

    def get_cache_file(self):
        with open(self.model_path,"rb") as f:
            bytes = f.read()
            readable_hash = hashlib.sha256(bytes).hexdigest()
        cache_file = "_".join(self.model_path.parts[self.model_path.parts.index("models") + 1:]) \
            + f"-{readable_hash}.pkl"
        cache_file = self.cache_dir / cache_file
        return cache_file

    def execute(self, model_input, attributes, identifying_attribute, force_single_value_attributes, mode: EngineMode):
        single_row = identifying_attribute is None

        cache_key = (tuple(model_input.data.index.unique(0)), torch.cuda.device_count())
        cached_result = self.check_cache(cache_key)
        result = cached_result if cached_result is not None else self.translate(model_input.bpe_file, single_row)
        self.update_cache(cache_key, cached_result is None, result)

        table_name = model_input.table_name[1].lower()
        result = {k.lower(): v for k, v in result.items()}.get(table_name, None)
        if result is None:
            return self.get_empty_result(model_input.data, attributes)

        if identifying_attribute is not None:
            _identifying_attribute = next(c for c in result.columns if c.lower() == identifying_attribute.lower())
            result = result.rename({_identifying_attribute: identifying_attribute}, axis=1)
            result[identifying_attribute] = result[identifying_attribute].apply(lambda x: x[0] if x != "" else x)
            result = result.groupby([result.index.name, identifying_attribute]) \
                .agg(lambda x: x.iloc[0]).reset_index(identifying_attribute)
        else:
            result = result.groupby(result.index.names).agg(lambda x: x.iloc[0])

        for a in attributes:
            if a not in result.columns:
                result[a] = ""

        if identifying_attribute is not None and identifying_attribute in model_input.evidence_columns:
            index_map = {v: k for k, v in enumerate(model_input.data.index.unique(0))}
            result = model_input.data.reset_index().replace({model_input.data.index.names[0]: index_map}).rename(
                {model_input.data.index.names[0]: "__idx__"}, axis=1).merge(result, on=(
                "__idx__", identifying_attribute), how="left").drop("__idx__", axis=1)[attributes].fillna("")
        return result

    def translate(self, bpe_text_file, single_row):
        old_dir = os.getcwd()
        os.chdir("text_to_table")
        with open(bpe_text_file) as f_in:
            with open(bpe_text_file.parent / "result", "w") as f_out:
                subprocess.run(["fairseq-interactive", str(bpe_text_file.parent / "bins"),
                                "--path", str(self.model_path),
                                "--beam", "5", "--remove-bpe",
                                "--buffer-size", "1024", "--max-tokens", "8192", "--max-len-b", "1024", "--user-dir",
                                "src/", "--task", "text_to_table_task", "--table-max-columns", "38"], stdin=f_in,
                                stdout=f_out)
        subprocess.run(["bash", "scripts/eval/convert_fairseq_output_to_text.sh", bpe_text_file.parent / "result"])
        os.chdir(old_dir)
        return self.get_pred_label_tables(bpe_text_file.parent / "result.text", single_row)

    def get_pred_label_tables(self, output_file, single_row):
        all_tables = []
        with open(output_file) as f:
            for i, translated_tables in enumerate(f):
                all_tables.append(self.convert_output_to_df(translated_tables.strip().replace(NEW_LINE, "\n"),
                                                            single_row))
                for k in all_tables[-1].keys():
                    all_tables[-1][k]["__idx__"] = i
        table_names = set(n for t in all_tables for n in t)
        all_tables = {n: pd.concat([t[n] for t in all_tables if n in t])
                      .fillna("").reset_index(drop=True).set_index("__idx__")
                      for n in table_names}
        return all_tables

    def convert_output_to_df(self, text, single_row):
        data = defaultdict(list)
        current_key = None
        for x in filter(bool, map(str.strip, text.split("\n"))):
            if not x.startswith("|"):
                current_key = x.strip(":")
            else:
                data[current_key].append(x)

        func = self.convert_text_to_df_single_row if single_row else self.convert_text_to_df_multi_row
        tables = {k: func("\n".join(v)) for k, v in data.items()}
        return tables
  
    def convert_text_to_df_single_row(self, text: str) -> pd.DataFrame:
        rows = text.split('\n')
        df_dict = dict()

        for i in range(0, len(rows)):

            entry_texts = self.convert_row_to_cells(rows[i])
            if len(entry_texts) != 2:
                continue
            if entry_texts[1]:
                df_dict[entry_texts[0]] = [[x.strip() for x in entry_texts[1].split(",")]]
            else:
                df_dict[entry_texts[0]] = [""]
        result = pd.DataFrame(df_dict)
        result.rename(columns={'': 'Name'}, inplace=True)
        return result


    def convert_text_to_df_multi_row(self, text: str) -> pd.DataFrame:
        rows = text.split('\n')
        header = rows[0]

        header_texts = self.convert_row_to_cells(header)
        df_dict = {}
        for text in header_texts:
            df_dict[text.strip()] = []

        for i in range(1, len(rows)):

            entry_texts = self.convert_row_to_cells(rows[i])
            for j, item in enumerate(df_dict.items()):
                if entry_texts[j]:
                    item[1].append([x.strip() for x in entry_texts[j].split(",")])
                else:
                    item[1].append("")
        result = pd.DataFrame(df_dict)
        result.rename(columns={'': 'Name'}, inplace=True)
        return result


    def convert_row_to_cells(self, row: str) -> list:
        """
        return array of cell texts
        """
        texts = row.split(SEPARATOR_STRING)
        left_offset = 0
        right_offset = 0
        if row.endswith(SEPARATOR_STRING):
            right_offset = 1
        if row.startswith(SEPARATOR_STRING):
            left_offset = 1
        return [x.strip() for x in texts[left_offset: -right_offset]]
