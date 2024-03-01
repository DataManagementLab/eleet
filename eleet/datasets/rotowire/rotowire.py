from pathlib import Path
import pandas as pd
from eleet.database import Database, Table, TextCollection, TextCollectionLabels


def load_rotowire_legacy(db_dir, split):
    # load json files
    result = dict()
    for file in (Path(db_dir) / "db" / "train").iterdir():
        name = file.name.split(".")[0]
        df = pd.read_json(file)
        result["train_" + name] = df.sort_index()  # type: ignore

    for file in (Path(db_dir) / "db" / split).iterdir():
        name = file.name.split(".")[0]
        df = pd.read_json(file)
        result[name] = df.sort_index()  #type: ignore

    # fix legacy
    team_map = dict(map(lambda x: tuple(x[1]), result["train_team_evidence"][["Team Name", "name"]].iterrows()))
    result = {
        k: v.replace(team_map)
            .drop("name", axis=1, errors="ignore")
            .rename({"Player Name": "name"}, axis=1, errors="ignore")
            .rename({"Team Name": "name"}, axis=1, errors="ignore")
        for k, v in result.items()
    }

    # Fix col order
    result["player_labels"] = result["player_labels"][result["train_player_labels"].columns]
    result["player_alignments"] = result["player_alignments"][["Name_Matched"] + result["train_player_labels"].columns.tolist()]
    result["team_labels"] = result["team_labels"][result["train_team_labels"].columns]
    result["team_alignments"] = result["team_alignments"][["Name_Matched"] + result["train_team_labels"].columns.tolist()]

    # set up tables and text collections
    player_stats = result["train_player_labels"].reset_index(drop=True)
    team_stats = result["train_team_labels"].reset_index(drop=True)
    player_stats["Game ID"] = player_stats["Game ID"] - player_stats["Game ID"].max() - 1
    team_stats["Game ID"] = team_stats["Game ID"] - team_stats["Game ID"].max() - 1
    tables = [
        Table(name="player_stats", data=player_stats, key_columns=["Game ID"]),
        Table(name="team_stats", data=team_stats, key_columns=["Game ID"]),
        Table(name="player_info", data=result["player_evidence"], key_columns=["name"]),
        Table(name="team_info", data=result["team_evidence"], key_columns=["name"]),
        Table(name="player_to_reports", data=result["player_to_reports"], key_columns=["name", "Game ID"]),
        Table(name="team_to_reports", data=result["team_to_reports"], key_columns=["name", "Game ID"])
    ]

    p_alignments = result["player_alignments"].set_index(["Game ID", "name"]).rename({"Name_Matched": "name"}, axis=1) \
            .applymap(lambda x: [] if len(x) == 0 else (x if isinstance(x[0], list) else [x]))
    p_normed = result["player_labels"].set_index(["Game ID", "name"], drop=False).drop("Game ID", axis=1) \
            .applymap(lambda x: [] if x == "" else [x]) * p_alignments.applymap(len)
    t_alignments = result["team_alignments"].set_index(["Game ID", "name"]).rename({"Name_Matched": "name"}, axis=1) \
            .applymap(lambda x: [] if len(x) == 0 else (x if isinstance(x[0], list) else [x]))
    t_normed = result["team_labels"].set_index(["Game ID", "name"], drop=False).drop("Game ID", axis=1) \
            .applymap(lambda x: [] if x == "" else [x]) * t_alignments.applymap(len)
    labels_player = TextCollectionLabels(
        normed=p_normed,
        alignments=p_alignments,
    )
    labels_team = TextCollectionLabels(
        normed=t_normed,
        alignments=t_alignments,
    )
    reports = TextCollection(name="reports", data=result["reports_table"], key_columns=["Game ID"])
    reports.setup_text_table("Player", attributes=result["player_labels"].columns.tolist(), multi_row=True,
                             labels=labels_player, identifying_attribute="name")
    reports.setup_text_table("Team", attributes=result["team_labels"].columns.tolist(), multi_row=True,
                             labels=labels_team, identifying_attribute="name")
    database = Database(
        name="rotowire", tables=tables, texts=[reports]
    )
    return database
