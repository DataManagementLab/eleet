from collections import namedtuple
from contextlib import ExitStack, contextmanager
import csv
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from attr import field
from eleet.methods.base_preprocessor import BasePreprocessor
from attrs import define


ModelInput = namedtuple("ModelInput", ["bpe_file", "data", "table_name", "evidence_columns"])


@define
class T2TPreprocessor(BasePreprocessor):
    encoder_json: str = field(converter=lambda x: str(Path(x).absolute()))
    vocab_bpe: str = field(converter=lambda x: str(Path(x).absolute()))

    @contextmanager
    def compute_model_input(self, data, report_column, report_table_name, extract_attributes, identifying_attribute,
                            example_rows, multi_table, limit=2**32, index_mode: bool=False):
        evidence_columns = self.get_evidence_columns(data, report_column)
        with TemporaryDirectory(dir=".", prefix="t2t_tmp") as tempdir:
            reports = data.data.reset_index()[[data.data.index.names[0], report_column]].drop_duplicates() \
                .sort_values(data.data.index.names[0]).drop(data.data.index.names[0], axis=1).applymap(str.strip) \
                .iloc[:limit]
            tempdir = Path(tempdir)
            text_file_path = tempdir / "reports.text"
            split, lang = tuple(text_file_path.name.split("."))

            with open(text_file_path, "w") as f:
                for _, text in reports.iterrows():
                    print(text.iloc[0].strip().replace("\n", " "), file=f)

            script_path = Path("text_to_table/scripts/multiprocessing_bpe_encoder.py").absolute()
            bart_dir = Path("bart.base").absolute()
            old_dir = os.getcwd()
            os.chdir(text_file_path.parent)

            subprocess.run(["python", str(script_path), "--encoder-json",
                            self.encoder_json, "--vocab-bpe", self.vocab_bpe,
                            "--inputs", f"{split}.{lang}", "--outputs",
                            f"{split}.bpe.{lang}", "--workers", "60", "--keep-empty"])

            subprocess.run(["fairseq-preprocess", "--source-lang", lang, "--target-lang", lang,
                            "--testpref", f"{split}.bpe", # "--validpref", "in.bpe", "--testpref", "in.bpe",
                            "--destdir", "bins/", "--workers", "60",
                            "--srcdict", f"{bart_dir}/dict.txt", "--tgtdict", f"{bart_dir}/dict.txt"])
            with open(f"{split}.bpe.text", "r") as f_i, open("tmp.bpe.text", "w") as f_o:
                for line in f_i:
                    line = line.split()
                    line = " ".join(line[:1023])
                    print(line, file=f_o)
            os.rename("tmp.bpe.text", f"{split}.bpe.text")

            os.chdir(old_dir)

            yield ModelInput((tempdir / f"{split}.bpe.text").absolute(),
                             data.data.iloc[data.data.index.get_level_values(0) < limit],
                             report_table_name,
                             evidence_columns)

    @contextmanager
    def compute_finetuning_data(self, data, split, limit=2**32):  # TODO rm limit
        single_row = all([d.identifying_attribute is None for d in data])
        data = sorted(data, key=lambda x: len(x.labels.normed))
        grouped_data = {d.full_name[0]: [dd for dd in data if dd.full_name[0] == d.full_name[0]] for d in data}
        with TemporaryDirectory(dir=".", prefix="t2t_tmp") as tempdir:
            with ExitStack() as stack:
                for group in grouped_data.values():
                    text_collection = group[0].text_collection
                    (f_text, f_data) = (
                        stack.enter_context(open(Path(tempdir) / f"{split}.{suffix}---{text_collection.name}", "w"))
                        for suffix in ("text", "data")
                    )
                    for i, text in text_collection.data.iterrows():
                        if i > limit:
                            break
                        func = self.serialize_for_t2t_single_row if single_row else self.serialize_for_t2t_multi_row
                        t2t_serialized = func(
                            rows=[d.labels.normed.loc[i] if i in d.labels.normed.index else None for d in group],
                            names=[d.name for d in group]
                        )
                        print(text.iloc[0].strip().replace("\n", " "), file=f_text)
                        print(t2t_serialized, file=f_data)
            yield tempdir

    def serialize_for_t2t_multi_row(self, rows, names):
        rows = [(r[[c for c in r.columns if c not in r.index.names]] if r is not None else r)
                for r in rows]
        use_index = [(r.index.dtype == "O") if r is not None else False for r in rows]
        result = list()
        for i, (name, use_idx) in enumerate(zip(names, use_index)):
            result.append(f"{name}:")
            sub = rows[i]
            if sub is None:
                continue
            selected_columns = [c for c in sub.columns if sub[c].apply(len).any()]
            header = " | ".join([""] + ([sub.index.name if sub.index.name != "name" else ""] if use_idx else [])
                                + selected_columns + [""]).strip()
            result.append(header)

            for lbl, row in sub.iterrows():
                values = [", ".join(sorted(set(row[c]))) for c in selected_columns]
                row_str = " | ".join([""] + ([lbl] if use_idx else []) + values + [""]).strip()
                result.append(row_str)
        return " <NEWLINE> ".join(result)
    
    def serialize_for_t2t_single_row(self, rows, names):
        result = list()
        for i, name in enumerate(names):
            result.append(f"{name}:")
            row = rows[i]
            if row is None:
                continue
            for col, val in row.iteritems():
                val = ", ".join(sorted(set(row[col])))
                if val:
                    result.append(f"| {col} | {val} |")
        return " <NEWLINE> ".join(result)
