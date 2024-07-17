from typing import List, Optional
from unidecode import unidecode
from collections import namedtuple
from functools import partial
import numpy as np
import pandas as pd
import logging

from tqdm import tqdm

from eleet_pretrain.metrics.metrics import compute_metrics_from_statistics
from fuzzywuzzy import fuzz

RunDescription = namedtuple("RunDescription", ["method", "query", "split_size", "dataset", "test", "limit"])
RunResult = namedtuple("RunResult", ["predictions", "labels", "text_index_name", "relevant_columns", "relevant_ids",
                                     "identifying_attribute", "runtime", "index_build_time", "metadata"])
logger = logging.getLogger(__name__)


def evaluate(collected_results):
    collected_stats = list()
    for k, result in collected_results.items():
        assert isinstance(k, RunDescription)
        print(f"Evaluate {k}")

        this_relevant_columns = [c for c in result.labels.data.columns if c in result.relevant_columns]
        aggregate_mode = any(isinstance(i, tuple) and i and isinstance(i[0], str) for i in result.labels.data.index)
        if aggregate_mode:
            this_labels = result.labels.data.reset_index()
            this_predictions = result.predictions.data.reset_index()
            this_labels[result.labels.data.index.name] = this_labels[result.labels.data.index.name] \
                .apply(lambda x: [set(x)])
            matched = match_identifying_attribute(this_predictions[[result.predictions.data.index.name]],
                                        this_labels[[result.labels.data.index.name]])
            pred_ids, label_ids = zip(*matched)
            this_predictions = this_predictions.loc[[i for i in pred_ids if i >= 0]].reset_index(drop=True)
            this_labels = this_labels.loc[[i for i in label_ids if i >= 0]].reset_index(drop=True)
        else:
            relevant_ids_labels = [r for r in result.relevant_ids
                                   if r in result.labels.data.index.get_level_values(0)]
            relevant_ids_preds = [r for r in result.relevant_ids
                                  if r in result.predictions.data.index.get_level_values(0)]
            this_predictions = result.predictions.data.loc[relevant_ids_preds][this_relevant_columns]
            this_labels = result.labels.data.loc[relevant_ids_labels][this_relevant_columns]
        stats = compute_stats(this_predictions, this_labels, k, result.identifying_attribute, result.metadata)
        collected_stats.append(stats)
    stats = pd.concat(collected_stats)
    metrics = compute_metrics(stats)
    runtimes = pd.DataFrame([{
        **{f: getattr(k, f) for f in k._fields},
        "runtime": v.runtime.total_seconds(),
        "index_build_time": v.index_build_time.total_seconds()
    } for k, v in collected_results.items()])
    return stats, metrics, runtimes


def compute_metrics(stats):
    result = dict()
    for group_id, group_df in stats.groupby(list(RunDescription._fields)):
        result[RunDescription(*group_id)] = compute_metrics_from_statistics(group_df, prefixes=[""])
    result = pd.DataFrame(result).T
    result.index.names = list(RunDescription._fields)
    return result


def compute_stats(pred, labels, desc, identifying_attribute, metadata):
    results = dict()
    string_labels = labels.applymap(lambda x: isinstance(x, str)).all()
    if any(string_labels):
        invalid_labels = labels.columns[string_labels].tolist()
        logger.warn(f"Skipping invalid labels columns {invalid_labels} for {desc}."
                    "Probably caused by join with overlapping columns.")
        labels = labels[[c for c in labels.columns if c not in invalid_labels]]
        pred = pred[[c for c in pred.columns if c not in invalid_labels]]

    for index_id in tqdm(pred.index.union(labels.index).unique(0), desc="-".join(map(str, desc[:4]))):
        tp, fp, fn = compute_counts(pred.loc[[index_id]] if index_id in pred.index else pred.iloc[:0],
                                    labels.loc[[index_id]] if index_id in labels.index else labels.iloc[:0],
                                    identifying_attribute=identifying_attribute)
        this_metadata = metadata.loc[index_id]
        results[(*desc, index_id)] = {"tp": tp, "fp": fp, "fn": fn, "id": index_id, **this_metadata}
    result = pd.DataFrame(results, index=["tp", "fp", "fn", "id", *metadata.columns]).T
    if len(result) == 0:
        result.index= pd.MultiIndex.from_arrays([[] for _ in range(len(RunDescription._fields) + 1)],
                                                names=[*RunDescription._fields, "idx"])  # type: ignore
    result[["tp", "fp", "fn"]] = result[["tp", "fp", "fn"]].astype(int)  # type: ignore
    result.index.names = [*RunDescription._fields, "idx"]
    return result


def compute_counts(pred_table, label_table, identifying_attribute):
    label_cols = [c for c in label_table.columns]
    pred_cols = [c for c in pred_table.columns if c in label_cols]
    pred_table = pred_table[pred_cols].copy()
    label_table = label_table[label_cols].copy()
    map_identifying_attribute(pred_table, label_table, identifying_attribute)

    tp = 0
    for i in sorted(set(label_table.index) & set(pred_table.index)):
        for col in sorted(set(label_table.columns) & set(pred_table.columns)):
            value = pred_table.loc[i][col]
            if isinstance(value, list):
                pred_values = [unidecode(str(v).lower().replace(" ", ""))
                               if isinstance(v, (str, int, float)) else unidecode(v[0].lower().replace(" ", ""))
                               for v in value if v != ""]
            else:
                pred_values: List[Optional[str]] = [str(value).lower().replace(" ", "")] if value != "" else []
            label_values = [{unidecode(str(x).lower().replace(" ", "")) for x in v} for v in label_table.loc[i][col]]

            for j, value in enumerate(pred_values):
                matches = [value in label_value for label_value in label_values]
                if any(matches):
                    match = matches.index(True)
                    tp += 1
                    pred_values[j] = None
                    label_values = [label_value for i, label_value in enumerate(label_values) if i != match]
            pred_table.loc[i, col] = [v for v in pred_values if v is not None]
            label_table.loc[i, col] = label_values

    fp = pred_table.applymap(lambda x: 1 if isinstance(x, str) else len(x)).sum().sum()
    fn = label_table.applymap(len).sum().sum()
    return tp, fp, fn


def map_identifying_attribute(pred_table, label_table, identifying_attribute):
    do_map = len(label_table.index.names) == 1 and (len(label_table) > 1 or len(pred_table) > 1) \
        and identifying_attribute is not None
    if not do_map:
        return

    mapping = match_identifying_attribute(pred_table[[identifying_attribute]], label_table[[identifying_attribute]])
    pred_idx, label_idx = map(partial(filter, lambda x: x > -1), tuple(zip(*mapping)))
    pred_table["__idx__"] = label_table["__idx__"] = 0
    pred_table.iloc[pred_idx, pred_table.columns == "__idx__"] = np.arange(len(pred_table))
    label_table.iloc[label_idx, pred_table.columns == "__idx__"] = np.arange(len(label_table))
    pred_table.set_index("__idx__", drop=True, inplace=True)
    label_table.set_index("__idx__", drop=True, inplace=True)


def match_identifying_attribute(pred_values, label_values):
    from eleet.methods.eleet.value import MMValue

    assert len(pred_values.columns) == 1
    pred_values = pred_values[pred_values.columns[0]]
    label_values = label_values[label_values.columns[0]]
    similarity = np.zeros((len(pred_values), len(label_values)))
    for i, pred_value in enumerate(pred_values):
        for j, label_value in enumerate(label_values):
            label_value = label_value[0] if len(label_value) else ""

            if isinstance(pred_value, MMValue):
                pred_value = pred_value[0][0]

            for label_alternative in label_value:
                similarity[i, j] = max(similarity[i, j], fuzz.ratio(pred_value.lower(), label_alternative.lower()))
    return matching_from_similarity(pred_values, label_values, similarity, warning_if_mismatched=False)


def matching_from_similarity(pred_values, label_values, similarity, warning_if_mismatched=True):
    matching = []
    pred_matched, label_matched = set(), set()
    while (similarity > -1).any():
        argmax = similarity.argmax()
        max_i  = argmax // len(label_values)
        max_j = argmax % len(label_values)
        matching.append((max_i, max_j))
        pred_matched.add(max_i)
        label_matched.add(max_j)
        similarity[max_i, :] = -1
        similarity[:, max_j] = -1

    for i in range(len(pred_values)):
        if i not in pred_matched:
            matching.append((i, -1))
            if warning_if_mismatched:
                logger.warning("Mismatched label: %s, %s", pred_values, label_values)
    for j in range(len(label_values)):
        if j not in label_matched:
            matching.append((-1, j))
            if warning_if_mismatched:
                logger.warning("Mismatched label: %s, %s", pred_values, label_values)
    return matching
