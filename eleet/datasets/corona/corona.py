"""
Access the corona dataset.

The corona dataset consists of the summaries of the RKI's daily situational reports about COVID-19:
https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Situationsberichte/Gesamt.html

The texts have been annotated with information about where they mention the structured values. The evaluation part of
the dataset contains a ground-truth for the matching process for different extractors.

Each entry of the dataset is a json file of the following structure:
{
    "id": "<id of the document>",
    "text": "<summary of the report>",
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

NAME = "corona"

ATTRIBUTES = [
    "date",  # date of the report
    "new_cases",  # number of new cases
    "new_deaths",  # number of new deaths
    "incidence",  # 7-day incidence
    "patients_intensive_care",  # number of people in intensive care
    "vaccinated",  # number of people that have been vaccinated at least once
    "twice_vaccinated"  # number of people that have been vaccinated twice
]

TEST_SET = ['2020-12-02.json', '2020-12-16.json', '2020-11-16.json',
           '2021-01-21.json', '2021-01-08.json', '2020-12-09.json',
           '2021-02-09.json', '2021-01-18.json', '2020-12-19.json',
           '2020-11-19.json', '2020-11-11.json', '2020-12-15.json',
           '2020-12-01.json', '2020-11-23.json', '2020-12-12.json',
           '2021-01-10.json', '2020-12-26.json', '2020-11-13.json',
           '2020-11-30.json', '2020-11-20.json', '2021-01-26.json',
           '2021-02-10.json', '2020-11-15.json', '2020-12-23.json',
           '2020-12-05.json', '2021-02-08.json', '2020-12-28.json',
           '2021-01-16.json', '2021-01-14.json', '2021-01-23.json']


def load_documents(db_dir, split):
    """
    Load the corona dataset.

    This method requires the .txt files in the "datasets/corona/documents/" folder.
    """
    dataset: List = []

    path = os.path.join(db_dir, "documents", "*.json")
    for file_path in map(Path, glob(path)):
        if (int(split == "test") + int(file_path.name in TEST_SET)) % 2:
            continue
        with open(file_path, encoding="utf-8") as file:
            dataset.append(json.loads(file.read()))
    return dataset


def load_corona(db_dir, split):
    docs = load_documents(db_dir, split)
    train_docs = load_documents(db_dir, "train")

    union_evidence = pd.DataFrame([{k: [m["mention"] for m in v]
                                    for k, v in x["mentions"].items() if v} for x in train_docs])[ATTRIBUTES]
    union_evidence = union_evidence.applymap(lambda x: x[0] if isinstance(x, list) and len(x) else "")
    union_evidence["report_number"] = np.arange(len(train_docs)) - len(train_docs) - 1
    union_evidence = Table(name="corona_stats", data=union_evidence, key_columns=["report_number"])

    normed = pd.DataFrame([{k: [m["mention"] for m in v]
                              for k, v in x["mentions"].items() if v} for x in docs])[ATTRIBUTES]
    normed = normed.applymap(lambda x: [] if not isinstance(x, list) and (x == "" or pd.isna(x)) else [max(x, key=len)])
    alignments = pd.DataFrame([{k: [(m["start"], m["start"] + m["length"]) for m in v]
                              for k, v in x["mentions"].items() if v} for x in docs])[ATTRIBUTES]
    alignments = alignments.applymap(lambda x: [] if not isinstance(x, list) and (x == "" or pd.isna(x)) \
                                     else [max(x, key=lambda y: y[1] - y[0])])

    texts = pd.DataFrame([(i, x["text"].replace("\r", "")) for i, x in enumerate(docs)],
                         columns=["report_number", "text"])

    normed.index = texts["report_number"]
    alignments.index = texts["report_number"]
    labels = TextCollectionLabels(normed=normed, alignments=alignments)
    reports = TextCollection(name="reports", data=texts, key_columns=["report_number"])
    reports.setup_text_table("summary", attributes=ATTRIBUTES, multi_row=False,
                             identifying_attribute=None, labels=labels,
                             force_single_value_attributes=ATTRIBUTES)
    db = Database(
        name="corona", tables=[union_evidence], texts = [reports]
    )
    return db

if __name__ == "__main__":
    db = load_corona("datasets/corona", "train")
    print(db)