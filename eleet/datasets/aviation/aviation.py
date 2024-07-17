"""
Access the aviation dataset.

The aviation dataset consists of the executive summaries of the NTSB Aviation Accident Reports:
https://www.ntsb.gov/investigations/AccidentReports/Pages/aviation.aspx

The texts have been annotated with information about where they mention the structured values. The evaluation part of
the dataset contains a ground-truth for the matching process for different extractors.

Each entry of the dataset is a json file of the following structure:
{
    "id": "<id of the document>",
    "text": "<executive summary of the report>",
    "mentions": {
        "<attribute name>": [
            {
            "mention": "<text of the mention>",
            "start": <position of the first character of the mention>,
            "length": <length of the mention>
            }    # for each mention of the attribute in the text
        ]  # for each attribute
    },
    "mentions_diff_value": {
        #  same as "mentions", but with mentions of the same attribute class (e.g. city) but not the desired value
    },
    "evaluation": {
        "<extractor name>": {
                "all_extractions": [
                    <json-serialized extraction>  # for all extractions by the extractor
                ],
                "mentions": {
                    "<attribute name>": [<indices of corresponding extractions>]  # for each attribute
                },
                "mentions_diff_value": {
                    "<attribute name>": [<indices of corresponding extractions>]  # for each attribute
                }
            }  # for some extractors
        }
    }  # for each document
"""
import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import List
import numpy as np

import pandas as pd

from eleet.database import Database, Table, TextCollection, TextCollectionLabels

logger = logging.getLogger(__name__)

NAME = "aviation"

EXTRACT_ATTRIBUTES = [
    "event_date",  # date of the event
    "location_city",  # city or place closest to the site of the event
    "location_state",  # state the city is located in
    "airport_code",  # code of the airport
    "airport_name",  # airport name
    "aircraft_damage",  # severity of the damage to the aircraft
    # "far_description",  # applicable regulation part or authority
    "weather_condition"  # weather conditions at the time of the event
]

EVIDENCE_ATTRIBUTES = [
    "air_carrier",  # name of the operator of the aircraft
    "aircraft_registration_number",  # registration number of the involved aircraft
    "aircraft_make",  # name of the aircraft's manufacturer
    "aircraft_model",  # alphanumeric aircraft model code
]


TEST_SET = ['AAB-01-02.json', 'AAB-02-01.json', 'AAR-05-02.json',
            'AAR-98-04.json', 'AAR-11-02.json', 'AAR-09-03.json',
            'AAR-04-04.json', 'AAR1702.json', 'AAR-14-04.json',
            'AAR-10-05.json', 'AAB-06-03.json', 'AAR-02-01.json',
            'AAB-00-01.json', 'AAB-02-03.json', 'AAR-04-03.json',
            'AAR-13-01.json', 'AAR-11-05.json', 'AAR-01-02.json',
            'AAR-10-01.json', 'AAR1801.json', 'AAB-02-05.json',
            'AAR-08-03.json', 'AAR-09-06.json', 'AAR-14-03.json',
            'AAR-07-06.json', 'AAR-06-01.json', 'AAB-06-05.json',
            'AAB-07-02.json', 'AAR-07-02.json', 'AAR-12-02.json']


def load_documents(db_dir, split):
    """
    Load the aviation dataset.

    This method requires the .txt files in the "<db_dir>/aviation/documents/" folder.
    """
    assert split in ("train", "test")
    dataset: List = []

    path = os.path.join(db_dir, "documents", "*.json")
    for file_path in map(Path, glob(path)):
        if (int(split == "test") + int(file_path.name in TEST_SET)) % 2:
            continue
        with file_path.open(encoding="utf-8") as file:
            dataset.append(json.load(file))
    return dataset


def load_aviation(db_dir, split):
    docs = load_documents(db_dir, split)
    train_docs = load_documents(db_dir, "train")

    union_evidence = pd.DataFrame([{k: [m["mention"] for m in v]
                                    for k, v in x["mentions"].items() if v} for x in train_docs])[EXTRACT_ATTRIBUTES]
    union_evidence = union_evidence.applymap(lambda x: x[0] if isinstance(x, list) and len(x) else "")
    union_evidence["report_number"] = np.arange(len(train_docs)) - len(train_docs) - 1

    report_numbers = np.arange(len(docs))
    evidence = pd.DataFrame([{k: v[0]["mention"]
                              for k, v in x["mentions"].items() if v} for x in docs])[EVIDENCE_ATTRIBUTES]
    normed = pd.DataFrame([{k: [m["mention"] for m in v]
                              for k, v in x["mentions"].items() if v} for x in docs])[EXTRACT_ATTRIBUTES]
    normed = normed.applymap(lambda x: [] if not isinstance(x, list) and (x == "" or pd.isna(x)) else [max(x, key=len)])
    alignments = pd.DataFrame([{k: [(m["start"], m["start"] + m["length"]) for m in v]
                              for k, v in x["mentions"].items() if v} for x in docs])[EXTRACT_ATTRIBUTES]
    alignments = alignments.applymap(lambda x: [] if not isinstance(x, list) and (x == "" or pd.isna(x)) \
                                     else [max(x, key=lambda y: y[1] - y[0])])
    evidence["report_number"] = report_numbers

    aircraft_to_reports = evidence.groupby(EVIDENCE_ATTRIBUTES).agg(list) \
        .reset_index().explode("report_number")[["aircraft_registration_number", "report_number"]]

    evidence.drop("report_number", axis="columns", inplace=True)

    texts = pd.DataFrame([(i, x["text"].replace("\r", "")) for i, x in enumerate(docs)],
                         columns=["report_number", "text"])
    normed.index = texts["report_number"]
    alignments.index = texts["report_number"]

    tables = [
        Table(name="aircraft", data=evidence, key_columns=["aircraft_registration_number"]),
        Table(name="aircraft_to_reports", data=aircraft_to_reports, key_columns=["aircraft_registration_number",
                                                                                 "report_number"]),
        Table(name="incidents", data=union_evidence, key_columns=["report_number"])
    ]
    labels = TextCollectionLabels(normed=normed, alignments=alignments)
    reports = TextCollection(name="reports", data=texts, key_columns=["report_number"])
    reports.setup_text_table("incident", attributes=EXTRACT_ATTRIBUTES, multi_row=False,
                             identifying_attribute=None, labels=labels,
                             force_single_value_attributes=EXTRACT_ATTRIBUTES)
    db = Database(
        name="aviation", tables=tables, texts = [reports]
    )
    return db


if __name__ == "__main__":
    db = load_aviation("datasets/aviation", "test")
    print(db)
