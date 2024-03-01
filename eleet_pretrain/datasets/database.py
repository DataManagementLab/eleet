import pandas as pd
from abc import ABC, abstractstaticmethod, abstractclassmethod


class DbDescription:
    def __init__(self, reports_id_column, reports_text_column, secondary_id_column):
        self.reports_id_column = reports_id_column
        self.reports_text_column = reports_text_column
        self.secondary_id_column = secondary_id_column


class DataBase(ABC):
    def __init__(self):
        self._table_names = []
        self._reports_tables = {}
        self._join_evidence_tables = {}
        self._union_evidence_tables = {}
        self._union_evidence_reports = {}
        self._key_columns = {}
        self._alignments = {}
        self._labels = {}
        self._normed = {}

    def register_evidence(self, *, table_name, reports_table, join_evidence, union_evidence, union_evidence_reports,
                          key_column, alignments, normed, labels):
        self._table_names.append(table_name)
        self._reports_tables[table_name] = reports_table  # reports_table
        self._join_evidence_tables[table_name] = join_evidence  # evidence table
        self._union_evidence_tables[table_name] = union_evidence  # statik tabelle aus dem trainings --> labels.json vom trainingsset
        self._key_columns[table_name] = key_column  # Game ID
        self._alignments[table_name] = alignments
        self._normed[table_name] = normed
        self._labels[table_name] = labels
        self._union_evidence_reports[table_name] = union_evidence_reports
        assert reports_table.index.name is not None
        assert join_evidence.index.names == alignments.index.names == labels.index.names == union_evidence.index.names \
            == normed.index.names == (self.reports_id_column(table_name), self.secondary_id_column(table_name))

    @abstractstaticmethod
    def load_db(self, db_path, split):
        pass

    @abstractclassmethod
    def process_db(self, *args, **kwargs):
        pass

    @property
    def table_names(self):
        return self._table_names

    def reports_table(self, table_name):
        return self._reports_tables[table_name]
    
    def key_column(self, table_name):
        return self._key_columns[table_name]

    def join_evidence(self, table_name):
        return self._join_evidence_tables[table_name]

    def union_evidence(self, table_name):
        return self._union_evidence_tables[table_name]
    
    def text_columns(self, table_name, include_key_column=False):
        if include_key_column:
            return self._union_evidence_tables[table_name].columns
        else:  
            return pd.Index([x for x in self._union_evidence_tables[table_name].columns
                             if x != self.key_column(table_name)])

    def description(self, table_name):
        return DbDescription(self.reports_id_column(table_name),
                             self.reports_text_column(table_name),
                             self.secondary_id_column(table_name))

    def union_evidence_reports(self, table_name):
        return self._union_evidence_reports[table_name]

    def labels(self, table_name, include_key_column):
        if include_key_column:
            return self._labels[table_name]
        else:
            return self._labels[table_name].drop(self.key_column(table_name), axis=1)

    def alignments(self, table_name, include_key_column):
        if include_key_column:
            return self._alignments[table_name]
        else:
            return self._alignments[table_name].drop(self.key_column(table_name), axis=1)
    
    def normed(self, table_name, include_key_column):
        if include_key_column:
            return self._normed[table_name]
        else:
            return self._normed[table_name].drop(self.key_column(table_name), axis=1)

    def reports_id_column(self, table_name):
        return self._reports_tables[table_name].index.name

    def reports_text_column(self, table_name):
        return self._reports_tables[table_name].columns[0]

    def secondary_id_column(self, table_name):
        return self._join_evidence_tables[table_name].index.names[1]


class RotowireDB(DataBase):

    @classmethod
    def process_db(cls, *, reports_table, player_to_reports, player_labels, team_alignments, team_labels,
                   team_to_reports, player_alignments, player_evidence, team_evidence,
                   train_reports_table, train_player_labels, train_team_labels,
                   train_team_evidence, train_player_evidence):
        reports_table.set_index("Game ID", inplace=True)
        train_reports_table.set_index("Game ID", inplace=True)
        result = cls()

        player_evidence.set_index("Player Name", inplace=True)
        train_player_evidence.set_index("Player Name", inplace=True)
        player_evidence.rename({"name": "Player Name"}, axis="columns", inplace=True)
        player_to_reports.set_index(["Game ID", "Player Name"], inplace=True)
        player_labels.set_index(["Game ID"], inplace=True)
        player_labels.set_index(["Player Name"], inplace=True, append=True, drop=False)
        train_player_labels.set_index(["Game ID"], inplace=True)
        train_player_labels.set_index(["Player Name"], inplace=True, append=True, drop=False)
        player_alignments.set_index(["Game ID", "Player Name"], inplace=True)

        result.register_evidence(table_name="player", reports_table=reports_table,
                                 join_evidence=player_evidence.join(player_to_reports),
                                 union_evidence=train_player_labels,
                                 union_evidence_reports=train_reports_table,
                                 alignments=player_alignments.join(player_to_reports).rename(
                                    {"Name_Matched": "Player Name"}, axis=1),
                                 normed=player_labels.join(player_to_reports),
                                 labels=player_labels.join(player_to_reports),
                                 key_column="Player Name")

        team_evidence.set_index("Team Name", inplace=True)
        train_team_evidence.set_index("Team Name", inplace=True)
        team_evidence.rename({"name": "Team Name"}, axis="columns", inplace=True)
        team_to_reports.set_index(["Game ID", "Team Name"], inplace=True)
        team_labels.set_index(["Game ID"], inplace=True)
        team_labels.set_index(["Team Name"], inplace=True, append=True, drop=False)
        train_team_labels.set_index(["Game ID"], inplace=True)
        train_team_labels.set_index(["Team Name"], inplace=True, append=True, drop=False)
        team_alignments.set_index(["Game ID", "Team Name"], inplace=True)

        result.register_evidence(table_name="team", reports_table=reports_table,
                                 join_evidence=team_evidence.join(team_to_reports),
                                 union_evidence=train_team_labels,
                                 union_evidence_reports=train_reports_table,
                                 alignments=team_alignments.join(team_to_reports).rename(
                                    {"Name_Matched": "Team Name"}, axis=1),
                                 normed=team_labels.join(team_to_reports),
                                 labels=team_labels.join(team_to_reports),
                                 key_column="Team Name")
        return result


    @staticmethod
    def load_db(db_dir, split):
        """Load the Rotowire database from disk."""
        result = dict()
        for file in (db_dir / "train").iterdir():
            name = file.name.split(".")[0]
            df = pd.read_json(file)
            result[name] = df

        result = {
            "train_reports_table": result["reports_table"],
            "train_player_labels": result["player_labels"],
            "train_team_labels": result["team_labels"],
            "train_team_evidence": result["team_evidence"],
            "train_player_evidence": result["player_evidence"],
        }
        for file in (db_dir / split).iterdir():
            name = file.name.split(".")[0]
            df = pd.read_json(file)
            result[name] = df
        return RotowireDB.process_db(**result)


class TRExDB(DataBase):

    @classmethod
    def process_db(cls, data):
        result = cls()
        for table in ("nobel-Personal", "nobel-Career", "countries-Geography",
                      "countries-Politics", "skyscrapers-Location", "skyscrapers-History"):
            id_name = "Entity ID"
            secondary_id_name = ""
            reports_table = data[f"{table}-reports"]
            reports_table.index.name = id_name
            labels = data[f"{table}-labels"]
            labels[secondary_id_name] = 0
            labels.index.name = id_name
            labels.set_index(secondary_id_name, append=True, inplace=True)
            normed = data[f"{table}-normed"]
            normed[secondary_id_name] = 0
            normed.index.name = id_name
            normed.set_index(secondary_id_name, append=True, inplace=True)
            alignment = data[f"{table}-alignment"]
            alignment[secondary_id_name] = 0
            alignment.index.name = id_name
            alignment.set_index(secondary_id_name, append=True, inplace=True)

            train_reports_table = data[f"train-{table}-reports"]
            train_reports_table.index.name = id_name
            train_labels = data[f"train-{table}-labels"]
            train_labels[secondary_id_name] = 0
            train_labels.index.name = id_name
            train_labels.set_index(secondary_id_name, append=True, inplace=True)

            result.register_evidence(table_name=table,
                                     reports_table=reports_table,
                                     join_evidence=labels.iloc[:0],
                                     union_evidence=train_labels,
                                     union_evidence_reports=train_reports_table,
                                     alignments=alignment,
                                     normed=normed,
                                     labels=labels,
                                     key_column="name")
        return result

    @staticmethod
    def load_db(db_dir, split):
        """Load the TREx database from disk."""
        print("Loading TREx database...")

        result = dict()
        for file in (db_dir / "train").iterdir():
            name = file.name.split(".")[0]
            df = pd.read_json(file)
            result[name] = df

        result = {
            f"train-{table}-{k}": result[f"{table}-{k}"]
            for k in ("reports", "labels", "normed", "alignment")
            for table in ("nobel-Personal", "nobel-Career", "countries-Geography",
                          "countries-Politics", "skyscrapers-Location", "skyscrapers-History")
        }
        for file in (db_dir / split).iterdir():
            name = file.name.split(".")[0]
            df = pd.read_json(file)
            result[name] = df
        return TRExDB.process_db(result)