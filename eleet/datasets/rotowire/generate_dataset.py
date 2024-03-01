from collections import defaultdict
import shutil
import pandas as pd
import numpy as np
import logging
import h5py
from copy import copy
from pathlib import Path
from contextlib import ExitStack
from transformers import BertTokenizerFast
from eleet_pretrain.datasets.database import RotowireDB
from eleet_pretrain.datasets.input_formatting.tabert_input_formatter import EleetTaBertInputFormatter
from eleet.datasets.rotowire.rotowire_dataset import Rotowire
from eleet.datasets.rotowire.align import Aligner
from eleet.datasets.rotowire.wiki_infoboxes import WikiInfoboxes
from eleet_pretrain.model.config import VerticalEleetConfig
from eleet_pretrain.datasets.base_loader import BaseLoader


logger = logging.getLogger()


SUBSAMPLE_BRACKETS = tuple(4 ** x for x in range(1, 20))


class CollectedData():
    def __init__(self, table_name, text):
        self.table_name = table_name
        self.text = text
        self.previous = None
        self.player_evidence_df = None
        self.player_alignment_df = None
        self.player_value_df = None
        self.team_evidence_df = None
        self.team_alignment_df = None
        self.team_value_df = None

    def set_player_dfs(self, evidence_df, alignment_df, value_df):
        self.player_evidence_df = evidence_df
        self.player_alignment_df = alignment_df
        self.player_value_df = value_df

    def set_team_dfs(self, evidence_df, alignment_df, value_df):
        self.team_evidence_df = evidence_df
        self.team_alignment_df = alignment_df
        self.team_value_df = value_df
    
    def add_previous(self, previous):
        self.previous.append(previous)
    
    def with_previous(self, previous):
        result = copy(self)
        result.previous = previous
        return result

    def __iter__(self):
        return iter(filter(lambda x: x[1] is not None, (
            ("Player", self.player_evidence_df, self.player_alignment_df, self.player_value_df, self.player_previous),
            ("Team", self.team_evidence_df, self.team_alignment_df, self.team_value_df, self.team_previous)
        )))

    @property
    def player_previous(self):
        return [{"table_name": p.table_name, "text":p.text, 
                 "evidence": p.player_evidence_df, "alignment": p.player_alignment_df, "values": p.player_value_df}
                for p in self.previous if p.player_evidence_df is not None]

    @property
    def team_previous(self):
        return [{"table_name": p.table_name, "text":p.text, 
                 "evidence": p.team_evidence_df, "alignment": p.team_alignment_df, "values": p.team_value_df}
                for p in self.previous if p.team_evidence_df is not None]

    @property
    def is_complete(self):
        return self.player_alignment_df is not None and self.team_alignment_df is not None and \
            self.team_alignment_df.shape[0] > 0 and self.player_alignment_df.shape[0] > 0

class GenerateDataset(BaseLoader):
    def __init__(self, data_dir, cache_dir, model_path=None):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.db_dir = self.cache_dir.parent / "db"
        self.aligner = Aligner(data_dir, cache_dir, model_path or Path(__file__).parent / "train.clf")
        self.wiki = WikiInfoboxes(data_dir, cache_dir)
        self.config = VerticalEleetConfig(max_num_cols=20)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.base_model_name)
        self.input_formatter = EleetTaBertInputFormatter(self.config, self.tokenizer)
        self.rng = np.random.default_rng()

    def generate_db(self):
        for split in ("train", "valid", "test"):
            split_dir = self.db_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            db = self._generate_db(split)
            for name, table in db.items():
                table.to_json(split_dir / f"{name}.json")
                

    def _generate_db(self, split="valid", max_num_samples=2**32):
        player_evidence, team_evidence = self.wiki.load_player_evidence_table(), self.wiki.load_team_evidence_table()
        team_evidence.index.name = "Team Name"
        player_evidence.index.name = "Player Name"
        rotowire = Rotowire(split=split, data_dir=self.data_dir)

        reports_table = list()
        player_to_reports = list()
        team_to_reports = list()
        player_labels = list()
        team_labels = list()
        player_alignments = list()
        team_alignments = list()
        alignments_path = self.get_alignments_path(split, self.data_dir)
        with open(alignments_path) as f:
            for _, line, (i, team_df, player_df, text) in zip(range(max_num_samples), f, rotowire):
                if "Name" in team_df:
                    team_to_reports += [(n, i) for n in team_df["Name"]]
                    assert len(team_evidence.loc[team_df["Name"]]) == len(team_df)
                if "Name" in player_df:
                    player_to_reports += [(n, i) for n in player_df["Name"]]
                    len(player_evidence.loc[player_df["Name"]]) == len(player_df)

                reports_table.append((i, text))
                player_alignment_df, team_alignment_df = tuple(pd.read_json(js) for js in line.split("\t"))
                player_alignment_df, team_alignment_df = self.rm_overlap(player_alignment_df, team_alignment_df)

                team_df["Game ID"] = player_df["Game ID"] = i
                player_alignment_df["Game ID" ] = team_alignment_df["Game ID"] = i

                team_labels.append(team_df)
                player_labels.append(player_df)
                player_alignments.append(player_alignment_df)
                team_alignments.append(team_alignment_df)
        
        reports_table = pd.DataFrame(reports_table, columns=["Game ID", "Report"])
        team_to_reports = pd.DataFrame(team_to_reports, columns=["Team Name", "Game ID"])
        player_to_reports = pd.DataFrame(player_to_reports, columns=["Player Name", "Game ID"])

        team_labels = pd.concat(team_labels, axis=0).rename({"Name": "Team Name"}, axis=1).fillna("")
        player_labels = pd.concat(player_labels, axis=0).rename({"Name": "Player Name"}, axis=1).fillna("")
        team_alignments = pd.concat(team_alignments, axis=0).rename({"Name": "Team Name"}, axis=1).fillna("")
        player_alignments = pd.concat(player_alignments, axis=0).rename({"Name": "Player Name"}, axis=1).fillna("")

        return {
            "reports_table": reports_table.reset_index(drop=True),
            "team_to_reports": team_to_reports.reset_index(drop=True),
            "player_to_reports": player_to_reports.reset_index(drop=True),
            "team_labels": team_labels.reset_index(drop=True),
            "player_labels": player_labels.reset_index(drop=True),
            "team_alignments": team_alignments.reset_index(drop=True),
            "player_alignments": player_alignments.reset_index(drop=True),
            "player_evidence": player_evidence.reset_index(),
            "team_evidence": team_evidence.reset_index()
        }

    def generate_training_set(self, out_path, max_num_samples=2**32):
        generation_funcs = (("multi_union", self.generate_multi_union), ("multi_join", self.generate_multi_join))
        write_mode = "w"
        for split in ("train", "valid"):
            db = RotowireDB.load_db(self.db_dir, split=split)
            for task_name, preprocess_func in generation_funcs:
                for dataset in ("team", "player"):
                    tensor_dict = preprocess_func(dataset, db, split)
                    split_name = f"{split}_{task_name}_{dataset}" if split != "train" else split
                    self.store_final_results(out_path, split=split_name,
                                            encodings=tensor_dict, write_mode=write_mode)
                    write_mode = "a"

    def generate_multi_join(self, dataset, db, split):
        preprocessor = JoinPreprocessor(config=self.config, tokenizer=self.tokenizer,
                                        db_description=db.description(dataset), max_query_columns=7)
        _, tensor_dict = preprocessor.generate_input(
            tabular_operand=db.join_evidence(dataset),
            textual_operand=db.reports_table(dataset),
            select_attributes=db.text_columns(dataset, include_key_column=False),
            key_column=db.key_column(dataset),
            alignments=db.alignments(dataset, include_key_column=True),
            labels=db.labels(dataset, include_key_column=False),
        )
        return tensor_dict

    def generate_multi_union(self, dataset, db, split):
        preprocessor = UnionPreprocessor(config=self.config, tokenizer=self.tokenizer,
                                         db_description=db.description(dataset),
                                         max_query_columns=11, pick_evidence_brackets=SUBSAMPLE_BRACKETS)
        tensor_dict = preprocessor.generate_input(
            tabular_operand=db.union_evidence(dataset),
            textual_operand=db.reports_table(dataset),
            evidence_reports=db.union_evidence_reports(dataset),
            key_column=db.key_column(dataset),
            alignments=db.alignments(dataset, include_key_column=True),
            labels=db.labels(dataset, include_key_column=True),
            is_train=True
        )
        return tensor_dict

    def generate_alignments(self):
        for split in ("train", "valid", "test"):
            with open(self.cache_dir / f"{split}-alignment.jsonl", "w") as f:
                dataset = Rotowire(split=split, data_dir=self.data_dir)
                for i, team_df, player_df, line_text in dataset:
                    alignment_player, alignment_team = self.aligner.align(player_df, team_df, line_text)
                    print(alignment_player.to_json() + "\t" + alignment_team.to_json(), file=f)

    def generate_training_set_subsets_ours(self, data_file, pick):
        split = "train"
        with h5py.File(data_file, "r+") as f:
            for split_size, mask in sorted(pick.items()):
                new_split = f"{split}.{split_size}"
                if new_split in f:
                    del f[new_split]
                f.create_group(new_split)
                for key in f[split]:
                    new_data = np.array(f[split][key])[mask]
                    f[new_split].create_dataset(key, shape=new_data.shape, data=new_data, compression="lzf")

    def generate_training_set_subsets_t2t(self, subset_dir, pick):
        split = "train"
        dataset = Rotowire(split=split, data_dir=self.data_dir)
        with ExitStack() as stack:
            file_handles = dict()
            for split_size in pick.keys():
                this_dir = (subset_dir / str(split_size))
                this_dir.mkdir(exist_ok=True, parents=True)
                file_handles[split_size] = {key: stack.enter_context(open(this_dir / (f"train.{key}"), "w"))
                                            for key in ("data", "text", "alignments")}
                for filename in ("test.text", "test.data", "valid.text", "valid.data"):
                    shutil.copy(self.data_dir / filename, this_dir)
            with open(self.cache_dir / f"{split}-alignment.jsonl", "r") as f:
                for i, ((line_data, line_text), alignment) in enumerate(zip(dataset.iter_raw(), f)):
                    for split_size in pick.keys():
                        if i >= split_size:
                            continue
                        print(line_data, file=file_handles[split_size]["data"], end="")
                        print(line_text, file=file_handles[split_size]["text"], end="")
                        print(alignment, file=file_handles[split_size]["alignments"], end="")

    def subsample_training_set(self, subset_dir, data_file):
        pick = dict()
        with h5py.File(data_file, "r") as f:
            table_ids = pd.Series(f["train"]["origin"][:, 0]).apply(
                lambda x: x.decode("ascii").split("-")[0]).astype(int).values
            unique_table_ids = np.unique(table_ids)
            num_training_tables = unique_table_ids.shape[0]
            split_sizes = list(filter(lambda s: s < num_training_tables, SUBSAMPLE_BRACKETS))
            if split_sizes[-1] != len(unique_table_ids):
                split_sizes.append(len(unique_table_ids))

            for size in split_sizes:
                mask = table_ids < size
                pick[size] = mask
        self.generate_training_set_subsets_t2t(subset_dir, pick)
        self.generate_training_set_subsets_ours(data_file, pick)

    def get_alignments_path(self, split, data_dir):
        alignments_path = data_dir / (split + ".alignments")
        if not alignments_path.exists():
            alignments_path = self.cache_dir / f"{split}-alignment.jsonl"
        return alignments_path

    def rm_overlap(self, *dfs):
        result = []
        for df in dfs:
            for col in df.columns:
                data = df[col]
                if not data.apply(lambda x: isinstance(x, list)).any():
                    continue

                nested_list = data.apply(lambda x: isinstance(x, list) and len(x) and isinstance(x[0], list)).any()
                if not nested_list:
                    data = data.apply(lambda x: [x] if isinstance(x, list) else x)
                data = self._rm_overlap(data)
                if not nested_list:
                    data = data.apply(lambda x: x[0])
                df[col] = data
                df[col] = df[col].fillna("")
                if nested_list:
                    df[col] = df[col].apply(lambda x: [] if x == "" else x)
            result.append(df)
        return result

    def _rm_overlap(self, data):
        name = data.name
        data = sorted([(i, *x) for i, d in data.iteritems() for x in d], key=lambda x: x[1])
        current_end = -1
        result_list = []
        for i, s, e in data:
            if s >= current_end:
                current_end = e
                result_list.append((i, s, e))
            elif result_list[-1][2] - result_list[-1][1] < e - s:
                result_list[-1] = (i, s, e)
                current_end = e
        result_dict = defaultdict(list)
        for i, s, e in result_list:
            result_dict[i].append([s, e])
        return pd.Series(result_dict, name=name)
