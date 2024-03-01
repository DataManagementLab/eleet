from contextlib import ExitStack
import shutil
import h5py
from eleet_pretrain.datasets.base_loader import BaseLoader
from transformers import BertTokenizerFast
import numpy as np
import pandas as pd
from eleet_pretrain.datasets.database import TRExDB
from eleet.datasets.trex.collected_data import TRExCollectedData
from eleet.datasets.trex.fixed_testset import FIXED_TESTSET
from eleet_pretrain.datasets.input_formatting.tabert_input_formatter import EleetTaBertInputFormatter

from eleet_pretrain.model.config import VerticalEleetConfig


NAMES = {
    "head of state": "Politics",
    "highest point": "Geography",
    "place of birth": "Personal",
    "occupation": "Career",
    "country": "Location",
    "date of official opening": "History"
}

SUBSAMPLE_BRACKETS = tuple(4 ** x for x in range(1, 20))


class TRExExtractor(BaseLoader):
    def __init__(self, *args, data_dir, cache_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.config = VerticalEleetConfig(max_num_cols=20)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.base_model_name)
        self.input_formatter = EleetTaBertInputFormatter(self.config, self.tokenizer)
        self.rng = np.random.default_rng()
        self.db_dir = cache_dir.parent / "db"

    def get_data(self):
        sub_datasets = ["nobel", "countries", "skyscrapers"]

        results = dict()
        for ds in sub_datasets:
            texts = pd.concat([n["texts"] for n in self.store.get_node(f"{ds}_default_union").nodes()])["text"]
            answers = pd.concat([n["answers"] for n in self.store.get_node(f"{ds}_default_union").nodes()])
            columns = pd.concat([n["header_columns"] for n in self.store.get_node(f"{ds}_default_union").nodes()])
            col_ids = pd.concat([n["header_column_ids"] for n in self.store.get_node(f"{ds}_default_union").nodes()])
            rows = pd.concat([n["rows"] for n in self.store.get_node(f"{ds}_default_union").nodes()])
            answers.loc[answers["answer_normalized"] == "", "answer_normalized"] = \
                answers.loc[answers["answer_normalized"] == "", "answer_surfaceform"]

            for (table_id, row_id), text in texts.iteritems():  # iter over all texts
                answers.sort_index(inplace=True)
                if (table_id, row_id) not in answers.index:
                    continue
                this_answers = answers.loc[table_id, row_id]  # data for current text
                this_columns = columns.loc[table_id]
                this_col_ids = col_ids.loc[table_id]
                row_normed_t2t = rows.loc[table_id, row_id].copy()
                row_alignment = rows.loc[table_id, row_id].copy()
                row_normed_ours = rows.loc[table_id, row_id].copy()
                for i in row_alignment.index:  # iterate over columns
                    col_id = this_col_ids.loc[i]
                    align = []
                    normed_ours = []
                    normed_t2t = ""
                    if col_id in this_answers.index:
                        normed_t2t = ", ".join(this_answers.loc[col_id]["answer_normalized"].unique())
                        align = this_answers.loc[col_id][["answer_start", "answer_end"]].apply(list, axis=1).tolist()
                        normed_ours = this_answers.loc[col_id]["answer_normalized"].tolist()
                        assert len(align) == len(normed_ours)
                    row_normed_t2t.loc[i] = normed_t2t
                    row_alignment.loc[i] = align
                    row_normed_ours.loc[i] = normed_ours
                row_normed_t2t.index = this_columns
                row_alignment.index = this_columns
                row_normed_ours.index = this_columns


                results[row_alignment.name[-1]] = results.get(row_alignment.name[-1], list())
                results[row_alignment.name[-1]].append((ds, row_normed_ours, row_normed_t2t, text, row_alignment))
        return results

    def split(self, data, valid_set_size=3):
        all_entity_ids = sorted(data.keys())
        test = [x for x in all_entity_ids if x in FIXED_TESTSET]
        other_ids = [x for x in all_entity_ids if x not in FIXED_TESTSET]
        self.rng.shuffle(other_ids)
        split_point = int(len(other_ids) * ((100 - valid_set_size) / 100))
        train = other_ids[: split_point]
        valid = other_ids[split_point:]
        return {"train": tuple(data[k] for k in train), "test": tuple(data[k] for k in test),
                "valid": (tuple(data[k] for k in valid))}

    def collect(self, splits):
        result = dict()
        for split_name, data in splits.items():
            split_result = dict()
            for d in data:
                dataset_name, row_normed_ours, row_normed_t2t, text, row_alignment = tuple(zip(*d))
                names = self.get_names(row_normed_t2t)

                assert all(dataset_name[0] == t for t in dataset_name)
                assert all(text[0] == t for t in text)
                text = text[0]
                dataset_name = dataset_name[0]

                for n, l_ours, l_t2t, a in zip(names, row_normed_ours, row_normed_t2t, row_alignment):
                    split_result[dataset_name, n] = split_result.get((dataset_name, n), list())
                    split_result[dataset_name, n].append((text, l_ours, l_t2t, a))
            result[split_name] = TRExCollectedData(split_result)
        return result


    def write(self, collected, data_dir=None, max_rows=2**32):
        data_dir = data_dir or self.data_dir
        for split_name, db in collected.items():
            with ExitStack() as stack:
                (f_text, f_data, f_align, f_normed) = (
                    stack.enter_context(open(data_dir / f"{split_name}.{suffix}", "w"))
                    for suffix in ("text", "data", "align", "normed")
                )
                for dataset_name in sorted(set(t[0] for t in db.reports_tables)):
                    tables = sorted([t[1] for t in db.reports_tables if t[0] == dataset_name])
                    for i, e_id in enumerate(db.reports_tables[dataset_name, tables[0]].index):
                        if i > max_rows:
                            continue
                        t2t_serialized = self.serialize_for_t2t(db.labels_tables, dataset_name, tables, e_id)
                        print(db.reports_tables[dataset_name, tables[0]].loc[e_id, "Report"], file=f_text)
                        print(t2t_serialized, file=f_data)
                        print("\t".join([db.alignments_tables[dataset_name, t].loc[e_id].to_json() for t in tables]),
                              file=f_align)
                        print("\t".join([db.normed_tables[dataset_name, t].loc[e_id].to_json() for t in tables]),
                              file=f_normed)

    def get_names(self, rows):
        this_names = [name for r in rows for col, name in NAMES.items() if col in r]
        assert len(this_names) == len(rows)
        return this_names

    def generate_db(self, out_dir):
        data = self.get_data()
        splits = self.split(data)
        collected = self.collect(splits)
        self.write(collected=collected)
        self._generate_db(collected=collected, out_dir=out_dir)

    def _generate_db(self, collected, out_dir):
        col_limit = 8
        for split in ("train", "valid", "test"):
            split_dir = out_dir / "db" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            db = collected[split]
            for name in db.labels_tables.keys():
                cols = db.alignments_tables[name].columns[:col_limit]  # TODO rm col limit
                db.alignments_tables[name][cols].to_json(split_dir / f"{'-'.join(name)}-alignment.json")
                db.reports_tables[name].to_json(split_dir / f"{'-'.join(name)}-reports.json")
                db.labels_tables[name][cols].to_json(split_dir / f"{'-'.join(name)}-labels.json")
                db.normed_tables[name][cols].to_json(split_dir / f"{'-'.join(name)}-normed.json")
