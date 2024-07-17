import sys

from openai import OpenAI

from eleet.datasets.diagnoses.generate import load_diagnoses
from eleet.methods.openai.engine import LLM_COST
sys.path = [x for x in sys.path if not x.endswith("eleet")]

import argparse
from collections import namedtuple
from glob import glob
import logging
import pickle
import re
import subprocess
import numpy as np
from pathlib import Path

from eleet.datasets.aviation.aviation import load_aviation
from eleet.datasets.corona.corona import load_corona
from eleet.datasets.rotowire.rotowire import load_rotowire_legacy
from eleet.datasets.trex.trex import load_trex_legacy
from eleet.evaluate import RunDescription, RunResult, evaluate
from eleet.methods.labels.engine import LabelEngine
from eleet.methods.labels.preprocessor import LabelPreprocessor
from eleet.methods.operator import Join, MMAggregation, MMJoin, MMSelection, MMUnion, MMScan, Projection, Selection



logger = logging.getLogger(__name__)


Query = namedtuple("Query", ["plan", "text_index_name", "relevant_columns", "relevant_ids", "identifying_attribute",
                             "all_split_sizes"])


LIMIT = 2 ** 32
OPENAI_METHODS = tuple(LLM_COST.keys())
METHODS = (
    "eleet",
    "eleet-no-vertical",
    "t2t",
    "llama",
    "bert",
    "tabert",
    "gpt-ft",
    "llama-ft"
) + OPENAI_METHODS

DATASETS = (
    "rotowire",
    "trex",
    "aviation",
    "corona",
    "diagnoses",
)

MODELS = {
    "eleet": "models/{dataset}/ours/finetuned/current/{split_size}/pytorch_model.bin",
    "eleet-no-vertical": "models/{dataset}/ours/finetuned/no-vertical/{split_size}/pytorch_model.bin",
    "bert": "models/{dataset}/ours/finetuned/bert/{split_size}/pytorch_model.bin",
    "tabert": "models/{dataset}/ours/finetuned/tabert/{split_size}/pytorch_model.bin",
    "t2t": "models/{dataset}/text_to_table/current/checkpoints.{^split_size}/checkpoint_average_best-3.pt",
    "llama-ft": "models/{dataset}/llama/current/checkpoint.{^split_size}/adapter_model.safetensors",
}
STORE_DIR = Path("predictions")


def get_model_path(method, dataset, split_size):
    template, options, split_sizes = _get_available_split_sizes(method, dataset)
    fuzzy = "{^split_size}" in template
    if split_size in split_sizes:
        return options[split_sizes.index(split_size)]
    diff = [split_size - s for s in split_sizes]
    diff = [(2 ** 32) if x < 0 else x for x in diff]

    model_path = options[np.argmin(diff)]

    # no warning if fuzzy and use requests largest split (full dataset)
    if not fuzzy or sum([d >= 0 for d in diff]) != len(diff):
        logger.warning(f"Could not find correct model for {method} on {dataset} for split size {split_size}. "
                       f"Using {model_path}. (Fuzzy={fuzzy})")
    return model_path


def _get_available_split_sizes(method, dataset):
    template = MODELS[method].replace("{dataset}", dataset)
    options = glob(re.sub(r"{[^}]*}", "*", template))
    regex = re.compile(re.escape(template).replace(r"\{\*\}", ".*")
                                          .replace(r"\{\^split_size\}", r"(\^?\d+)")
                                          .replace(r"\{split_size\}", r"(\^?\d+)"))
    split_sizes = [int(regex.match(o)[1]) for o in options]  # type: ignore
    split_sizes, options = zip(*sorted(zip(split_sizes, options)))
    return template, options, split_sizes


def get_available_split_sizes(method, dataset, split_sizes):
    split_sizes = {str(x) for x in split_sizes}
    if method == "gpt-ft":
        client = OpenAI()
        models = client.models.list()
        selected_split_sizes = sorted(set(
            [int(m.id.split(":")[-2].split("-")[1])
             for m in models.data if m.id.startswith("ft:") and m.id.split(":")[-2].split("-")[0] == dataset]
        ))
    elif method not in MODELS:
        selected_split_sizes = (4, 16, 64, 256, 1024, 4096)
    else:
        _, _, selected_split_sizes = _get_available_split_sizes(method, dataset)
    result = [x for x in selected_split_sizes if split_sizes == {"all"} or str(x) in split_sizes]
    result = sorted(result, key=lambda x: -2 ** 32 if isinstance(x, str) else -x)
    return result


def get_queries(datasets, test, limit):
    funcs = [
        globals()[f"get_{dataset}_queries"]
        for dataset in datasets
    ]
    for f in funcs:
        db, db_train, queries = f(test=test, limit=limit)
        for q in queries:
            yield db, db_train, q


def get_rotowire_queries(test, limit):
    db_train = load_rotowire_legacy(Path(__file__).parents[1] / "datasets" / "rotowire", "train")
    db = load_rotowire_legacy(Path(__file__).parents[1] / "datasets" / "rotowire", "test" if test else "valid")

    ## Player ##
    query_plan_player_join = MMJoin(operands=[
        Join(operands=["player_info", "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit)
    query_plan_player_union = MMUnion(operands=[
        "player_stats",
        "reports.Player",
    ], limit=limit)
    query_plan_player_scan = MMScan(operands=[
        "reports.Player",
    ], limit=limit)

    # Project
    PROJECT_ATTRS_PLAYER = ["Game ID", "name", "Points", "Assists", "Steals"]
    query_plan_player_join_project = MMJoin(operands=[
        Join(operands=["player_info", "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit, project_columns=PROJECT_ATTRS_PLAYER)
    query_plan_player_union_project = MMUnion(operands=[
        Projection(operands=["player_stats"], project_columns=PROJECT_ATTRS_PLAYER),
        "reports.Player",
    ], limit=limit)
    query_plan_player_scan_project = MMScan(operands=[
        "reports.Player",
    ], limit=limit, project_columns=PROJECT_ATTRS_PLAYER)

    # Select Table
    query_plan_player_join_select_table_001 = MMJoin(operands=[
        Join(operands=[Selection(operands=["player_info"], selectivity=0.01), "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit)
    query_plan_player_join_select_table_005 = MMJoin(operands=[
        Join(operands=[Selection(operands=["player_info"], selectivity=0.05), "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit)
    query_plan_player_join_select_table_010 = MMJoin(operands=[
        Join(operands=[Selection(operands=["player_info"], selectivity=0.10), "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit)
    query_plan_player_join_select_table_020 = MMJoin(operands=[
        Join(operands=[Selection(operands=["player_info"], selectivity=0.20), "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit)
    query_plan_player_join_select_table_030 = MMJoin(operands=[
        Join(operands=[Selection(operands=["player_info"], selectivity=0.30), "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit)
    query_plan_player_join_select_table_050 = MMJoin(operands=[
        Join(operands=[Selection(operands=["player_info"], selectivity=0.50), "player_to_reports"], join_key="name"),
        "reports.Player",
    ], join_key="Game ID", limit=limit)

    # Select Text
    query_plan_player_scan_select_text_001 = MMScan(operands=[
        MMSelection(operands=["reports.Player"], limit=limit, selectivity=0.01, attribute="Points"),
    ], limit=limit)
    query_plan_player_scan_select_text_005 = MMScan(operands=[
        MMSelection(operands=["reports.Player"], limit=limit, selectivity=0.05, attribute="Points"),
    ], limit=limit)
    query_plan_player_scan_select_text_010 = MMScan(operands=[
        MMSelection(operands=["reports.Player"], limit=limit, selectivity=0.10, attribute="Points"),
    ], limit=limit)
    query_plan_player_scan_select_text_020 = MMScan(operands=[
        MMSelection(operands=["reports.Player"], limit=limit, selectivity=0.20, attribute="Points"),
    ], limit=limit)
    query_plan_player_scan_select_text_030 = MMScan(operands=[
        MMSelection(operands=["reports.Player"], limit=limit, selectivity=0.30, attribute="Points"),
    ], limit=limit)
    query_plan_player_scan_select_text_050 = MMScan(operands=[
        MMSelection(operands=["reports.Player"], limit=limit, selectivity=0.50, attribute="Points"),
    ], limit=limit)

    # Aggregation
    query_plan_player_aggregation = MMAggregation(operands=[
        MMScan(operands=["reports.Player"], limit=limit)
    ], attribute="name")
    query_plan_player_aggregation_and_project = MMAggregation(operands=[
        MMScan(operands=["reports.Player"], limit=limit, project_columns=PROJECT_ATTRS_PLAYER)
    ], attribute="name")
    query_plan_player_aggregation_and_select = MMAggregation(operands=[
        MMScan(operands=[MMSelection(operands=["reports.Player"], limit=limit, selectivity=0.10, attribute="Points")],
               limit=limit)
    ], attribute="name")

    ## TEAM ##
    query_plan_team_join = MMJoin(operands=[
        Join(operands=["team_info", "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit)
    query_plan_team_union = MMUnion(operands=[
        "team_stats",
        "reports.Team",
    ], limit=limit)
    query_plan_team_scan = MMScan(operands=[
        "reports.Team",
    ], limit=limit)

    # Project
    PROJECT_ATTRS_TEAM = ["Game ID", "name", "Wins", "Losses", "Total points"]
    query_plan_team_join_project = MMJoin(operands=[
        Join(operands=["team_info", "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit, project_columns=PROJECT_ATTRS_TEAM)
    query_plan_team_union_project = MMUnion(operands=[
        Projection(operands=["team_stats"], project_columns=PROJECT_ATTRS_TEAM),
        "reports.Team",
    ], limit=limit)
    query_plan_team_scan_project = MMScan(operands=[
        "reports.Team",
    ], limit=limit, project_columns=PROJECT_ATTRS_TEAM)

    # Select Table
    query_plan_team_join_select_table_001 = MMJoin(operands=[
        Join(operands=[Selection(operands=["team_info"], selectivity=0.01), "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit)
    query_plan_team_join_select_table_005 = MMJoin(operands=[
        Join(operands=[Selection(operands=["team_info"], selectivity=0.05), "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit)
    query_plan_team_join_select_table_010 = MMJoin(operands=[
        Join(operands=[Selection(operands=["team_info"], selectivity=0.10), "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit)
    query_plan_team_join_select_table_020 = MMJoin(operands=[
        Join(operands=[Selection(operands=["team_info"], selectivity=0.20), "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit)
    query_plan_team_join_select_table_030 = MMJoin(operands=[
        Join(operands=[Selection(operands=["team_info"], selectivity=0.30), "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit)
    query_plan_team_join_select_table_050 = MMJoin(operands=[
        Join(operands=[Selection(operands=["team_info"], selectivity=0.50), "team_to_reports"], join_key="name"),
        "reports.Team",
    ], join_key="Game ID", limit=limit)


    # Select Text
    query_plan_team_scan_select_text_001 = MMScan(operands=[
        MMSelection(operands=["reports.Team"], limit=limit, selectivity=0.01, attribute="Total points"),
    ], limit=limit)
    query_plan_team_scan_select_text_005 = MMScan(operands=[
        MMSelection(operands=["reports.Team"], limit=limit, selectivity=0.05, attribute="Total points"),
    ], limit=limit)
    query_plan_team_scan_select_text_010 = MMScan(operands=[
        MMSelection(operands=["reports.Team"], limit=limit, selectivity=0.10, attribute="Total points"),
    ], limit=limit)
    query_plan_team_scan_select_text_020 = MMScan(operands=[
        MMSelection(operands=["reports.Team"], limit=limit, selectivity=0.20, attribute="Total points"),
    ], limit=limit)
    query_plan_team_scan_select_text_030 = MMScan(operands=[
        MMSelection(operands=["reports.Team"], limit=limit, selectivity=0.30, attribute="Total points"),
    ], limit=limit)
    query_plan_team_scan_select_text_050 = MMScan(operands=[
        MMSelection(operands=["reports.Team"], limit=limit, selectivity=0.50, attribute="Total points"),
    ], limit=limit)

    # Aggregation
    query_plan_team_aggregation = MMAggregation(operands=[
        MMScan(operands=["reports.Team"], limit=limit)
    ], attribute="name")
    query_plan_team_aggregation_and_project = MMAggregation(operands=[
        MMScan(operands=["reports.Team"], limit=limit, project_columns=PROJECT_ATTRS_TEAM)
    ], attribute="name")
    query_plan_team_aggregation_and_select = MMAggregation(operands=[
        MMScan(operands=[MMSelection(operands=["reports.Team"], limit=limit, selectivity=0.10, attribute="Total points")],
               limit=limit)
    ], attribute="name")

    RELEVANT_IDS_PLAYER = db.texts["reports"].text_tables["Player"].data.index
    RELEVANT_ATTRS_PLAYER = db.texts["reports"].text_tables["Player"].attributes
    RELEVANT_ATTRS_TEAM = db.texts["reports"].text_tables["Team"].attributes
    RELEVANT_IDS_TEAM = db.texts["reports"].text_tables["Team"].data.index
    return db, db_train, (
        Query(query_plan_player_join, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),
        Query(query_plan_player_union, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),
        Query(query_plan_player_scan, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),

        Query(query_plan_player_join_project, "Game ID", PROJECT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_union_project, "Game ID", PROJECT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_scan_project, "Game ID", PROJECT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),

        Query(query_plan_player_join_select_table_001, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_join_select_table_005, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_join_select_table_010, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),
        Query(query_plan_player_join_select_table_020, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_join_select_table_030, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_join_select_table_050, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),

        Query(query_plan_player_scan_select_text_001, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_scan_select_text_005, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_scan_select_text_010, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),
        Query(query_plan_player_scan_select_text_020, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_scan_select_text_030, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_scan_select_text_050, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),

        Query(query_plan_player_aggregation, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", True),
        Query(query_plan_player_aggregation_and_project, "Game ID", PROJECT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),
        Query(query_plan_player_aggregation_and_select, "Game ID", RELEVANT_ATTRS_PLAYER, RELEVANT_IDS_PLAYER, "name", False),

        Query(query_plan_team_join, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_union, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_scan, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),

        Query(query_plan_team_join_project, "Game ID", PROJECT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_union_project, "Game ID", PROJECT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_scan_project, "Game ID", PROJECT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),

        Query(query_plan_team_join_select_table_001, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_join_select_table_005, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_join_select_table_010, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_join_select_table_020, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_join_select_table_030, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_join_select_table_050, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),

        Query(query_plan_team_scan_select_text_001, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_scan_select_text_005, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_scan_select_text_010, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_scan_select_text_020, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_scan_select_text_030, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_scan_select_text_050, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),

        Query(query_plan_team_aggregation, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_aggregation_and_project, "Game ID", PROJECT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
        Query(query_plan_team_aggregation_and_select, "Game ID", RELEVANT_ATTRS_TEAM, RELEVANT_IDS_TEAM, "name", False),
    )


def get_trex_queries(test, limit):
    db_train = load_trex_legacy(Path(__file__).parents[1] / "datasets" / "trex", "train")
    db = load_trex_legacy(Path(__file__).parents[1] / "datasets" / "trex", "test" if test else "valid")

    union_nobel_personal = MMUnion(operands=[
        "nobel-Personal-union",
        "nobel_reports.Personal",
    ], limit=limit)

    union_nobel_career = MMUnion(operands=[
        "nobel-Career-union",
        "nobel_reports.Career",
    ], limit=limit)

    # Join
    join_nobel_personal = MMJoin(operands=[
        "nobel-Career-join",
        "nobel_reports.Personal",
    ], join_key="index", limit=limit)

    join_nobel_career = MMJoin(operands=[
        "nobel-Personal-join",
        "nobel_reports.Career",
    ], join_key="index", limit=limit)

    # Scan
    scan_nobel_personal = MMScan(operands=[
        "nobel_reports.Personal",
    ], limit=limit)

    scan_nobel_career = MMScan(operands=[
        "nobel_reports.Career",
    ], limit=limit)

    # Projection
    PROJECT_ATTRS_NOBEL_PERSONAL = ['index', 'name', 'place of birth', 'country of citizenship']
    union_nobel_personal_project = MMUnion(operands=[
        Projection(operands=["nobel-Personal-union"], project_columns=PROJECT_ATTRS_NOBEL_PERSONAL),
        "nobel_reports.Personal",
    ], limit=limit)

    PROJECT_ATTRS_NOBEL_CAREER = ['index', 'name', 'award received', 'educated at']
    union_nobel_career_project = MMUnion(operands=[
        Projection(operands=["nobel-Career-union"], project_columns=PROJECT_ATTRS_NOBEL_CAREER),
        "nobel_reports.Career",
    ], limit=limit)

    # Selection
    union_nobel_personal_selection = MMUnion(operands=[
        "nobel-Personal-union",
        MMSelection(operands=["nobel_reports.Personal"], limit=limit, selectivity=0.8, attribute="country of citizenship"),
    ], limit=limit)

    union_nobel_personal_selection2 = MMUnion(operands=[
        "nobel-Personal-union",
        MMSelection(operands=["nobel_reports.Personal"], limit=limit, selectivity=0.3, attribute="country of citizenship"),
    ], limit=limit)

    # Aggregation
    scan_nobel_personal_aggregation = MMAggregation(operands=[MMScan(operands=[
        "nobel_reports.Personal",
    ], limit=limit)], attribute="country of citizenship")

    scan_nobel_career_aggregation = MMAggregation(operands=[MMScan(operands=[
        "nobel_reports.Career",
    ], limit=limit)], attribute="field of work")


    def get_relevant_cols_ids(a, b, project_attrs=None):
        if project_attrs is None:
            return db.texts[f"{a}_reports"].text_tables[b].attributes, \
                db.texts[f"{a}_reports"].text_tables[b].data.index
        else:
            return project_attrs, \
                db.texts[f"{a}_reports"].text_tables[b].data.index

    return db, db_train, (
        Query(union_nobel_personal, "index", *get_relevant_cols_ids("nobel", "Personal"), None, False),
        Query(union_nobel_career, "index", *get_relevant_cols_ids("nobel", "Career"), None, False),

        Query(join_nobel_personal, "index", *get_relevant_cols_ids("nobel", "Personal"), None, False),
        Query(join_nobel_career, "index", *get_relevant_cols_ids("nobel", "Career"), None, False),

        Query(scan_nobel_personal, "index", *get_relevant_cols_ids("nobel", "Personal"), None, False),
        Query(scan_nobel_career, "index", *get_relevant_cols_ids("nobel", "Career"), None, False),

        Query(union_nobel_personal_selection, "index", *get_relevant_cols_ids("nobel", "Personal"), None, False),
        Query(union_nobel_personal_selection2, "index", *get_relevant_cols_ids("nobel", "Personal"), None, False),

        Query(union_nobel_personal_project, "index", *get_relevant_cols_ids("nobel", "Personal", PROJECT_ATTRS_NOBEL_PERSONAL), None, False),
        Query(union_nobel_career_project, "index", *get_relevant_cols_ids("nobel", "Career", PROJECT_ATTRS_NOBEL_CAREER), None, False),

        Query(scan_nobel_personal_aggregation, "index", *get_relevant_cols_ids("nobel", "Personal"), None, False),
        Query(scan_nobel_career_aggregation, "index", *get_relevant_cols_ids("nobel", "Career"), None, False),
    )

def get_corona_queries(test, limit):  # type: ignore
    assert test
    db_train = load_corona(Path(__file__).parents[1] / "datasets" / "corona", "train")
    db = load_corona(Path(__file__).parents[1] / "datasets" / "corona", "test")

    union = MMUnion(operands=["corona_stats", "reports.summary"], limit=limit)  # model only pre-trained on unions

    # rng = np.random.default_rng(42); x = set(
    #     tuple(sorted(rng.choice(db.texts["reports"].text_tables["summary"].attributes, 3, replace=False),
    #                  key=db.texts["reports"].text_tables["summary"].attributes.index)) for _ in range(15))

    PROJECTIONS = [
        ('report_number', 'new_cases', 'new_deaths', 'vaccinated'),
        ('report_number', 'date', 'patients_intensive_care', 'twice_vaccinated'),
        ('report_number', 'date', 'incidence', 'vaccinated'),
        ('report_number', 'date', 'new_cases', 'new_deaths'),
        ('report_number', 'new_deaths', 'vaccinated', 'twice_vaccinated'),
        ('report_number', 'date', 'vaccinated', 'twice_vaccinated'),
        ('report_number', 'new_deaths', 'incidence', 'vaccinated'),
        ('report_number', 'new_cases', 'new_deaths', 'incidence'),
        ('report_number', 'date', 'new_cases', 'patients_intensive_care', 'vaccinated'),
        ('report_number', 'new_cases', 'new_deaths', 'incidence', 'twice_vaccinated'),
        ('report_number', 'date', 'new_cases', 'new_deaths', 'patients_intensive_care'),
        ('report_number', 'date', 'new_cases', 'vaccinated', 'twice_vaccinated'),
        ('report_number', 'date', 'new_deaths', 'patients_intensive_care', 'twice_vaccinated'),
        ('report_number', 'date', 'new_deaths', 'incidence', 'vaccinated')
    ]
    # project
    union_projections = list()
    for projection in PROJECTIONS:
        union_projections.append(MMUnion(operands=[Projection(operands=["corona_stats"],
                                         project_columns=projection), "reports.summary"], limit=limit))

    RELEVANT_ATTRS = db.texts["reports"].text_tables["summary"].attributes
    RELEVANT_IDS = db.texts["reports"].text_tables["summary"].data.index
    return db, db_train, (
        Query(union, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        *(Query(q, "report_number", proj, RELEVANT_IDS, None, False) for q, proj in zip(union_projections, PROJECTIONS))
    )

def get_aviation_queries(test, limit):
    assert test
    db_train = load_aviation(Path(__file__).parents[1] / "datasets" / "aviation", "train")
    db = load_aviation(Path(__file__).parents[1] / "datasets" / "aviation", "test")

    # basic
    join = MMJoin(operands=[
        Join(operands=["aircraft", "aircraft_to_reports"],
             join_key="aircraft_registration_number"),
        "reports.incident"], join_key="report_number", limit=limit
    )
    union = MMUnion(operands=[
        "incidents", "reports.incident"
    ], limit=limit)
    scan = MMScan(operands=[
        "reports.incident"],
    limit=limit)

    # projection
    PROJECT_COLUMNS = ["report_number", "location_city", "location_state"]
    join_project = MMJoin(operands=[
        Join(operands=["aircraft", "aircraft_to_reports"],
             join_key="aircraft_registration_number"),
        "reports.incident"], join_key="report_number", limit=limit, project_columns=PROJECT_COLUMNS
    )
    union_project = MMUnion(operands=[
        Projection(["incidents"], project_columns=PROJECT_COLUMNS),
        "reports.incident"
    ], limit=limit)
    scan_project = MMScan(operands=[
        "reports.incident"],
    limit=limit, project_columns=PROJECT_COLUMNS)

    # Selection Table
    join_select_table_08 = MMJoin(operands=[
        Join(operands=[Selection(["aircraft"], selectivity=0.8), "aircraft_to_reports"],
             join_key="aircraft_registration_number"),
        "reports.incident"], join_key="report_number", limit=limit
    )
    join_select_table_03 = MMJoin(operands=[
        Join(operands=[Selection(["aircraft"], selectivity=0.3), "aircraft_to_reports"],
             join_key="aircraft_registration_number"),
        "reports.incident"], join_key="report_number", limit=limit
    )
    join_select_table_05 = MMJoin(operands=[
        Join(operands=[Selection(["aircraft"], selectivity=0.5), "aircraft_to_reports"],
             join_key="aircraft_registration_number"),
        "reports.incident"], join_key="report_number", limit=limit
    )

    # Selection Text
    union_select_text_08 = MMUnion(operands=[
        "incidents",
        MMSelection(operands=["reports.incident"], selectivity=0.8, attribute="aircraft_damage", limit=limit)
    ], limit=limit)
    union_select_text_03 = MMUnion(operands=[
        "incidents",
        MMSelection(operands=["reports.incident"], selectivity=0.3, attribute="location_state", limit=limit)
    ], limit=limit)
    union_select_text_05 = MMUnion(operands=[
        "incidents",
        MMSelection(operands=["reports.incident"], selectivity=0.5, attribute="weather_condition", limit=limit)
    ], limit=limit)

    # Aggregation
    scan_agg1 = MMAggregation(operands=[MMScan(operands=["reports.incident"], limit=limit)],
                              attribute="aircraft_damage")
    scan_agg2 = MMAggregation(operands=[MMScan(operands=["reports.incident"], limit=limit)],
                              attribute="location_state")
    scan_agg3 = MMAggregation(operands=[MMScan(operands=["reports.incident"], limit=limit)],
                              attribute="weather_condition")

    # TODO Join, Scan
    RELEVANT_ATTRS = db.texts["reports"].text_tables["incident"].attributes
    RELEVANT_IDS = db.texts["reports"].text_tables["incident"].data.index
    return db, db_train, (
        Query(join, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(union, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(scan, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),

        Query(join_project, "report_number", PROJECT_COLUMNS, RELEVANT_IDS, None, False),
        Query(union_project, "report_number", PROJECT_COLUMNS, RELEVANT_IDS, None, False),
        Query(scan_project, "report_number", PROJECT_COLUMNS, RELEVANT_IDS, None, False),

        Query(join_select_table_03, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(join_select_table_05, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(join_select_table_08, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),

        Query(union_select_text_03, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(union_select_text_05, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(union_select_text_08, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),

        Query(scan_agg1, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(scan_agg2, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
        Query(scan_agg3, "report_number", RELEVANT_ATTRS, RELEVANT_IDS, None, False),
    )


def get_diagnoses_queries(test, limit):
    assert test
    db_train = load_diagnoses(Path(__file__).parents[1] / "datasets" / "diagnoses", "train")
    db_valid = load_diagnoses(Path(__file__).parents[1] / "datasets" / "diagnoses", "test")
    union_health = MMUnion(operands=["health_issues", "reports.health_issues_new"], limit=limit)
    union_computer = MMUnion(operands=["computer_problems", "reports.computer_problems_new"], limit=limit)

    RELEVANT_ATTRS_HEALTH = db_valid.texts["reports"].text_tables["health_issues_new"].attributes
    RELEVANT_IDS_HEALTH = db_valid.texts["reports"].text_tables["health_issues_new"].data.index
    RELEVANT_ATTRS_COMPUTER = db_valid.texts["reports"].text_tables["computer_problems_new"].attributes
    RELEVANT_IDS_COMPUTER = db_valid.texts["reports"].text_tables["computer_problems_new"].data.index

    return db_valid, db_train, (
        Query(union_health, "report_number", RELEVANT_ATTRS_HEALTH, RELEVANT_IDS_HEALTH, "name", False),
        Query(union_computer, "report_number", RELEVANT_ATTRS_COMPUTER, RELEVANT_IDS_COMPUTER, "name", False),
    )


def get_methods(db, methods, db_train, allowed_split_sizes, store_dir):

    # ELEET #############################################################################################################
    method_strs = methods
    methods = set(m.split(":")[0] for m in methods)

    for x in ("eleet", "bert", "tabert", "eleet-no-vertical"):
        if x in methods:
            from eleet_pretrain.model.config import VerticalEleetConfig
            from eleet.methods.multi_modal_db.engine import ELEETEngine
            from eleet.methods.multi_modal_db.preprocessor import ELEETPreprocessor
            from transformers import BertTokenizerFast

            for finetune_split_size in get_available_split_sizes(x, db.name, allowed_split_sizes):
                additional_configs = {} if x != "eleet-no-vertical" else {"disable_vertical_transform": True}
                config = VerticalEleetConfig(max_num_cols=20, **additional_configs)
                tokenizer = BertTokenizerFast.from_pretrained(config.base_model_name)
                preprocessor = ELEETPreprocessor(config=config, tokenizer=tokenizer)
                engine = ELEETEngine(
                    model_name_or_path=get_model_path(x, db.name, finetune_split_size),
                    config=config, tokenizer=tokenizer, cache_dir=store_dir / "cache", name=x.upper()
                )
                yield preprocessor, engine, finetune_split_size
                del engine

    # TEXT TO TABLE ####################################################################################################

    if "t2t" in methods:
        from eleet.methods.text_to_table.preprocessor import T2TPreprocessor
        from eleet.methods.text_to_table.engine import T2TEngine

        for finetune_split_size in get_available_split_sizes("t2t", db.name, allowed_split_sizes):
            preprocessor = T2TPreprocessor(encoder_json=Path("datasets") / "rotowire" / "data" / "encoder.json",
                                           vocab_bpe=Path("datasets") / "rotowire" / "data" / "vocab.bpe")
            engine = T2TEngine(model_path=get_model_path("t2t", db.name, finetune_split_size),  # type: ignore
                               cache_dir=store_dir / "cache")
            yield preprocessor, engine, finetune_split_size
            del engine

    # GPT ##############################################################################################################

    for method_str in method_strs:
        method = method_str.split(":")[0]
        if method not in (set(OPENAI_METHODS) & methods):
            continue

        from eleet.methods.openai.engine import OpenAIEngine
        from eleet.methods.llama.preprocessor import LLMPreprocessor

        try:
            llm_batch_size = int(method_str.split(":")[1])
        except IndexError:
            llm_batch_size = 1

        for finetune_split_size in get_available_split_sizes(method, db.name, allowed_split_sizes):
            preprocessor = LLMPreprocessor(train_db=db_train, num_samples=finetune_split_size,
                                           finetune_split_size=finetune_split_size, llm_batch_size=llm_batch_size)
            engine = OpenAIEngine(llm=method, cache_dir=store_dir / "cache", llm_batch_size=llm_batch_size)
            # engine.enable_caching()
            yield preprocessor, engine, finetune_split_size
            del engine

    if "gpt-ft" in methods:
        from eleet.methods.openai.engine import OpenAIEngine
        from eleet.methods.llama.preprocessor import LLMPreprocessor
        from openai import OpenAI

        for finetune_split_size in get_available_split_sizes("gpt-ft", db.name, allowed_split_sizes):
            preprocessor = LLMPreprocessor(train_db=db_train, num_samples=0, finetune_split_size=0)

            client = OpenAI()
            models = client.models.list()
            model = max(tuple(
                [m
                 for m in models.data if m.id.startswith("ft:") and m.id.split(":")[-2] == f"{db.name}-{finetune_split_size}"]
            ), key=lambda x: x.created)


            engine = OpenAIEngine(llm=model.id, cache_dir=store_dir / "cache")
            # engine.enable_caching()
            yield preprocessor, engine, finetune_split_size
            del engine

    # LLaMA ############################################################################################################

    if "llama" in methods:
        from eleet.methods.llama.engine import LLaMAEngine
        from eleet.methods.llama.preprocessor import LLMPreprocessor

        for finetune_split_size in get_available_split_sizes("llama", db.name, allowed_split_sizes):
            preprocessor = LLMPreprocessor(train_db=db_train, num_samples=8, finetune_split_size=finetune_split_size)
            engine = LLaMAEngine(ckpt_dir="/mnt/labstore/SIGs/ML/llama-2-7b-chat/",
                                 tokenizer_path="/home/murban/llama/tokenizer.model",
                                 cache_dir=store_dir / "cache")
            yield preprocessor, engine, finetune_split_size
            del engine

    if "llama-ft" in methods:
        from eleet.methods.llama.engine import LLaMAEngine
        from eleet.methods.llama.preprocessor import LLMPreprocessor

        for finetune_split_size in get_available_split_sizes("llama-ft", db.name, allowed_split_sizes):
            preprocessor = LLMPreprocessor(train_db=db_train, num_samples=0, finetune_split_size=0)
            engine = LLaMAEngine(ckpt_dir=get_model_path("llama-ft", db.name, finetune_split_size),
                                 tokenizer_path="/home/murban/llama/tokenizer.model",
                                 cache_dir=store_dir / "cache")
            yield preprocessor, engine, finetune_split_size
            del engine


def print_queries(datasets, limit, only_flagged, skip_flagged, **_):
    print("\\begin{enumerate}")
    for db, _, query in get_queries(datasets=datasets, test=True, limit=limit):
        if only_flagged and not query.all_split_sizes:
            continue
        if skip_flagged and query.all_split_sizes:
            continue
        print(query.plan.latex(db))
    print("\\end{enumerate}")

def run_benchmark(datasets, methods, store_dir, execute, force_execute, split_sizes, test, limit, only_flagged,
                  skip_flagged, cost_estimation):
    collected_results = dict()
    label_preprocessor = LabelPreprocessor()
    label_engine = LabelEngine(cache_dir=store_dir / "cache")
    cached_results = load_cached_results(store_dir)
    exceptions = list()

    for db, db_train, query in get_queries(datasets=datasets, test=test, limit=limit):
        if only_flagged and not query.all_split_sizes:
            logger.warning(f"Skipping {query}, because it is not flagged.")
            continue
        if skip_flagged and query.all_split_sizes:
            logger.warning(f"Skipping {query}, because it is flagged.")
            continue

        labels = db.execute_query(query_plan=query.plan, preprocessor=label_preprocessor, engine=label_engine)
        text_metadata = db.get_text_metadata(query.plan)

        for preprocessor, engine, finetune_split_size in get_methods(methods=methods, db=db, db_train=db_train,
                                                                     allowed_split_sizes=split_sizes,
                                                                     store_dir=store_dir):
            print(f"Running {labels.name} with {engine.name}.")
            run_description = RunDescription(method=engine.name, query=labels.name, split_size=finetune_split_size,
                                             dataset=db.name, test=test, limit=limit)
            try:
                if cost_estimation:
                    engine.only_cost_estimation = True  # type: ignore

                if (execute and run_description not in cached_results) or force_execute:
                    result, runtime, index_build_time = db.execute_query(
                        query_plan=query.plan, preprocessor=preprocessor, engine=engine, measure_runtime=True)

                    if not cost_estimation:
                        store_result(store_dir, run_description, result, runtime, index_build_time)
                else:
                    if run_description not in cached_results:
                        logger.warning(f"No cached result for {run_description}")
                        continue
                    logger.warning(f"Use cached result for {run_description}")
                    result, runtime, index_build_time = cached_results[run_description]

                if result.name != labels.name:
                    logger.warning(f"Label and Predictions have different names: {result.name} != {labels.name}")

                collected_results.update(
                    {run_description: RunResult(
                        predictions=result, labels=labels,
                        text_index_name=query.text_index_name,
                        relevant_columns=query.relevant_columns,
                        relevant_ids=query.relevant_ids,
                        identifying_attribute=query.identifying_attribute,
                        runtime=runtime, index_build_time=index_build_time,
                        metadata=text_metadata
                    )}
                )

            except Exception as e:
                    logger.warn(e, exc_info=True)
                    exceptions.append((run_description, e))

    if exceptions or cost_estimation:
        for e in exceptions:
            print(e)
        return

    stats, metrics, runtimes = evaluate(collected_results)
    suffix = "-".join(tuple(methods) + tuple(datasets)
                      + tuple((split_sizes,) if isinstance(split_sizes, str) else split_sizes)
                      + ("test" if test else "valid", "limit", str(limit)))
    stats.to_csv(store_dir / f"stats_{suffix}.csv")
    runtimes.to_csv(store_dir / f"runtimes_{suffix}.csv")
    metrics.to_csv(store_dir / f"metrics_{suffix}.csv")
    return stats, metrics, runtimes


def store_result(store_dir, run_description, result, runtime, index_build_time):
    store_dir.mkdir(exist_ok=True, parents=True)
    name = "_".join(map(str, run_description)).replace(" ", "-") + ".pkl"
    with (store_dir / name).open("wb") as f:
        pickle.dump((run_description, result, runtime, index_build_time), f)


def load_cached_results(store_dir):
    result = {}
    if store_dir.exists():
        for file in store_dir.iterdir():
            if not file.name.endswith(".pkl"):
                continue
            with file.open("rb") as f:
                run_description, this_result, runtime, index_build_time = pickle.load(f)
                result[run_description] = (this_result, runtime, index_build_time)
    return result


def run_slurm(args,
              interfering_methods=("gpt-ft", ) + OPENAI_METHODS,
              cpu_models=("gpt", "gpt4", "gpt-ft") + OPENAI_METHODS,
              torchrun_models=("llama",),
              accelerate_models=("llama-ft", ),
              special_run_scripts=("llama-ft",)):

    fixed_args = ["--store-dir", str(args.store_dir),
                  "--limit", str(args.limit),
                  *(["--skip-execute"] if not args.execute else []),
                  *(["--force-execute"] if args.force_execute else []),
                  *(["--use-test-set"] if args.test else []),
                  *(["--only-flagged"] if args.only_flagged else []),
                  *(["--skip-flagged"] if args.skip_flagged else []),
                  *(["--cost-estimation"] if args.cost_estimation else [])]


    port_offset = 0
    for method_str in args.methods:
        method = method_str.split(":")[0]
        suffix = ""
        if method in special_run_scripts:
            suffix += f"-{method}"
        if method in cpu_models:
            suffix += "-cpu"
        run_script = f"slurm/run{suffix}.slurm"

        debug_run_command = ["python", "-m", "debugpy", "--listen", "5687", "--wait-for-client"]
        run_command = ["python"] if not args.debug else debug_run_command
        if method in torchrun_models:
            run_command = (["torchrun"] if not args.debug
                           else (debug_run_command + ["-m", "torch.distributed.launch"])) \
                + ["--nproc_per_node=2", "--rdzv_endpoint=localhost:{port}", "--rdzv_backend=c10d"]
        elif method in accelerate_models:
            # not allowed to set port to 0 for accelerate
            if args.port == 0:
                args.port = 29550
            run_command = (["accelerate", "launch"] if not args.debug
                           else (debug_run_command + ["-m", "accelerate.commands.launch"])) \
                + ["--main_process_ip", "127.0.0.1", "--main_process_port", "{port}"]
                # + ["--debug"]

        if method in interfering_methods:
            this_run_command = [c.replace("{port}", str(0 if args.port == 0 else args.port + port_offset))
                           for c in run_command]
            port_offset += 1
            sbatch_args = ["sbatch", run_script, *this_run_command, "eleet/benchmark.py",
                           "--methods", method_str,
                           "--datasets", *args.datasets,
                           "--split-sizes", *map(str, args.split_sizes),
                           *fixed_args]
            print(sbatch_args)
            subprocess.run(sbatch_args)
        else:
            for dataset in args.datasets:
                for split_size in args.split_sizes:
                    this_run_command = [c.replace("{port}", str(0 if args.port == 0 else args.port + port_offset))
                                   for c in run_command]
                    port_offset += 1
                    sbatch_args = ["sbatch", run_script, *this_run_command, "eleet/benchmark.py",
                                    "--methods", method_str,
                                    "--datasets", dataset,
                                    "--split-sizes", split_size,
                                    *fixed_args]
                    print(sbatch_args)
                    subprocess.run(sbatch_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm-mode", action="store_true")
    parser.add_argument("--methods", nargs="+", default=("eleet",),
                        help=f"methods to benchmark. Choose from {METHODS}.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS,
                        help="datasets to run benchmark on")
    parser.add_argument("--store-dir", type=Path, default=STORE_DIR, help="where to store predictions")
    parser.add_argument("--skip-execute", action="store_false", dest="execute",
                        help="print a warning when cached result is not available")
    parser.add_argument("--force-execute", action="store_true",
                        help="whether to load cached results if available")
    parser.add_argument("--split-sizes", nargs="+", default=["all"], help="which split sizes to evaluate")
    parser.add_argument("--use-test-set", action="store_true", dest="test", help="whether to use the test or valid set")
    parser.add_argument("--only-flagged", action="store_true",
                        help="only execute those queries that are flagged as to be evaluated on all split sizes")
    parser.add_argument("--skip-flagged", action="store_true",
                        help="skip those queries that are flagged as to be evaluated on all split sizes")
    parser.add_argument("--limit", type=int, default=LIMIT, help="limit dataset size")
    parser.add_argument("--cost-estimation", action="store_true", help="if set, only do cost estimation")
    parser.add_argument("--debug", action="store_true",
                        help="if set, run debugpy on port 5687. Only applies to slurm mode.")
    parser.add_argument("--port", type=int, default=0,
                        help="Specify the port for distrubuted inference (e.g. accelerate or torchrun)."
                        "Only applies to slurm mode.")
    parser.add_argument("--print-queries", action="store_true",
                        help="Print the queries that will be executed")
    args = parser.parse_args()

    if (args.split_sizes == ["all"] or len(args.split_sizes) > 1) and not args.only_flagged and not args.print_queries:
        raise ValueError("Are you sure you want to run ALL queries on multiple split sizes? Consider --only-flagged.")
    if args.skip_flagged and args.only_flagged:
        raise ValueError("You cannot set --skip-flagged and --only-flagged at the same time.")

    assert all(x.split(":")[0] in METHODS for x in args.methods), f"Methods must be one of {METHODS}"

    if args.slurm_mode:
        run_slurm(args)
        exit(0)

    kwargs = dict(datasets=args.datasets, methods=args.methods,
                  store_dir=args.store_dir / ("test" if args.test else "valid"),
                  execute=args.execute, force_execute=args.force_execute, split_sizes=args.split_sizes,
                  test=args.test, limit=args.limit, only_flagged=args.only_flagged, skip_flagged=args.skip_flagged,
                  cost_estimation=args.cost_estimation)
    print(kwargs)
    if args.print_queries:
        print_queries(**kwargs)
    else:
        run_benchmark(**kwargs)

