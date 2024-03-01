"""Module for computing metrics and statistics."""

import functools
import numpy as np
from numpy.lib import math
import pandas as pd


def compute_metrics(eval_pred, logging_path, split):
    """Compute the metrics."""
    predictions, labels = eval_pred
    
    statistics = compute_statistics(predictions, labels)
    for s in statistics:
        s.to_hdf(logging_path / "prediction_statistics.h5", key=f"{split}/{'_'.join(s.index.names)}", mode="a")
    summary = pd.concat([s.groupby(s.index.names[:-1]).sum() if "col_id" in s.index.names else s
                         for s in statistics], axis=1, sort=True)
    summary[summary.isna()] = 0

    metrics = compute_metrics_from_statistics(summary)
    return metrics


def compute_statistics(predictions, labels):
    """Compute False/True Positives/Negatives and store them in da DataFrame."""
    query_preds, query_labels, mlm_preds, mlm_labels, sample_ids, table_ids, row_ids, col_ids = get_preds_and_labels(
        predictions, labels)
    dfs = list()
    dfs.append(compute_token_level_query_statistics(query_preds, query_labels))
    dfs.append(compute_answer_level_query_statistics(query_preds, query_labels))
    result = pd.concat(dfs, axis=1)
    result.index = pd.MultiIndex.from_arrays([sample_ids, table_ids, row_ids, col_ids],
                                             names=["sample_id", "table_id", "row_id", "col_id"])

    if mlm_preds is not None:
        df_mlm = compute_mlm_statistics(mlm_preds, mlm_labels)
        y, x = np.unique(sample_ids, return_index=True)
        sample_ids = np.repeat(y, mlm_labels.shape[1])
        table_ids = np.repeat(table_ids[x], mlm_labels.shape[1])
        row_ids = np.tile(np.arange(mlm_labels.shape[1]), mlm_labels.shape[0])
        df_mlm.index = pd.MultiIndex.from_arrays([sample_ids, table_ids, row_ids],
                                                 names=["sample_id", "table_id", "row_id"])
        # dfs.append(compute_token_level_query_statistics(schema_preds, query_labels, query_to_table_id, prefix="sc_"))
        return result, df_mlm
    return result,


def get_preds_and_labels(predictions, labels):
    """Get the slice of active predictions and labels."""
    mlm_preds, mlm_labels = None, None
    if len(predictions) >= 5:
        query_preds, mlm_preds, idx, table_id = predictions[:4]
        query_labels, query_coords, mlm_labels = labels
    else:
        query_preds, idx, table_id = predictions[:3]
        query_labels, query_coords = labels
    query_mask = query_coords[:, :, 0] >= 0
    query_labels = query_labels[query_mask]
    query_preds = query_preds[query_mask]
    # if schema_preds is not None:
    #     schema_preds = schema_preds[query_mask]
    sample_ids = np.repeat(idx, query_mask.sum(1))
    table_ids = np.repeat(table_id, query_mask.sum(1), axis=0)
    table_ids = np.apply_along_axis(
        lambda x: np.array(["-".join(map(str, x))], dtype="object"), 1, table_ids).reshape(-1)
    row_ids, col_ids = query_coords[query_mask][:, :2].T
    return query_preds, query_labels, mlm_preds, \
        mlm_labels, sample_ids, table_ids, row_ids, col_ids


def compute_answer_level_query_statistics(predictions, labels, prefix="a_"):
    predicted_answers = get_answers(predictions)
    true_answers = get_answers(labels)
    result = np.zeros((predictions.shape[0], 3), int)
    for i, (t, p) in enumerate(zip(true_answers, predicted_answers)):
        tp = len(t & p)
        fp = len(p - t)
        fn = len(t - p)
        result[i] = np.array([tp, fp, fn])
    df = pd.DataFrame(result, columns=(f"{prefix}{x}" for x in ("tp", "fp", "fn")))
    return df

def get_answers(tensor):
    result = list()
    for q_id in range(tensor.shape[0]):
        answers = set()
        for t_id in np.where(tensor[q_id] == 2)[0]:
            answer = [t_id]
            for t in range(t_id + 1, tensor.shape[1]):
                if tensor[q_id, t] == 1:
                    answer.append(t)
                else:
                    break
            answers.add("-".join(map(str, answer)))  # TODO use input ids to get answer tokens
        result.append(answers)
    return result

def compute_token_level_query_statistics(predictions, labels, prefix="sd_"):
    """Compute False/True Positives/Negatives for token classification."""
    cols = list()
    for label in (1, 2):
        tp = np.sum(np.logical_and(predictions == label, labels == label), axis=1)
        fp = np.sum(np.logical_and(predictions == label, labels != label), axis=1)
        fn = np.sum(np.logical_and(predictions != label, labels == label), axis=1)
        cols.extend([tp, fp, fn])

    data = np.vstack(cols).T
    data = data[:, :3] + data[:, 3:]

    df1 = pd.DataFrame(
        data,
        columns=[f"{prefix}{x}" for x in ("tp", "fp", "fn")]
    )
    return df1

def compute_mlm_statistics(predictions, labels):
    """Compute False/True Positives/Negatives for token classification."""
    correct = np.sum(((predictions == labels) & (labels != -1)).reshape(-1, predictions.shape[-1]), -1)
    incorrect = np.sum(((predictions != labels) & (labels != -1)).reshape(-1, predictions.shape[-1]), -1)

    df1 = pd.DataFrame(
        [correct, incorrect],
        index=["mlm_true", "mlm_false"]
    ).T
    return df1

def compute_metrics_from_statistics(summary, prefixes=("sd_", "sc_", "a_")):
    """Compute the metrics."""
    collect = list()

    for prefix in prefixes:
        tp, fp, fn = f"{prefix}tp", f"{prefix}fp", f"{prefix}fn"
        if tp not in summary or fp not in summary or fn not in summary:
            continue
        mask = summary[[tp, fp, fn]].sum(1) > 0
        metrics = summary.loc[mask].copy()
        metrics_sum = metrics.sum()
        metrics_sum.name = "micro"
        metrics = pd.concat((metrics, pd.DataFrame(metrics_sum).T))
        precision_score = metrics[tp] / (metrics[tp] + metrics[fp] + 0.00000001)
        recall_score = metrics[tp] / (metrics[tp] + metrics[fn] + 0.00000001)
        precision_score[pd.isna(precision_score)] = 0
        recall_score[pd.isna(recall_score)] = 0
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score + 0.00000001)
        f1_score[pd.isna(f1_score)] = 0
        metrics[f"{prefix}precision"] = precision_score
        metrics[f"{prefix}recall"] = recall_score
        metrics[f"{prefix}f1"] = f1_score
        collect.append(metrics[[f"{prefix}precision", f"{prefix}recall", f"{prefix}f1"]])


    if "mlm_true" in summary:
        mask = summary["mlm_false"] + summary["mlm_true"] > 0
        metrics = summary.loc[mask].copy()
        metrics_sum = metrics.sum()
        metrics_sum.name = "micro"
        metrics = pd.concat((metrics, pd.DataFrame(metrics_sum).T))
        metrics["mlm_accuracy"] = metrics["mlm_true"] / (metrics["mlm_false"] + metrics["mlm_true"])
        collect.append(metrics[["mlm_accuracy"]])

    aggs = {
        "mean": pd.DataFrame.mean,
        "median": pd.DataFrame.median,
        "q1": functools.partial(pd.DataFrame.quantile, q=0.25), 
        "q3": functools.partial(pd.DataFrame.quantile, q=0.75),
        "min": pd.DataFrame.min,
        "max": pd.DataFrame.max
    }

    result = dict()
    for metrics in collect:
        result.update({
            f"{agg_name}_{metric_name}": value
            for agg_name, values in ((agg_name, agg_func(metrics.iloc[:-1], axis=0))
                                     for agg_name, agg_func in aggs.items())
            for metric_name, value in values.items()
        })
        result.update({f"micro_{k}": v for k, v in metrics.loc["micro"].T.items()})
    return result
