"""Train the model."""

import logging
import h5py
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy

import torch

from eleet_pretrain.utils import get_date_prefix, get_git_arg_hashes

logger = logging.getLogger(__name__)


AGG_LEVELS = (
    ("table_id", "sample_id"),
    ("table_id", "row_id", "sample_id"),
    ("table_id", "col_id", "sample_id"),
    ("table_id", "row_id", "col_id", "sample_id"),
    ("table_id", ),
    ("table_id", "row_id"),
    ("table_id", "col_id"),
    ("table_id", "row_id", "col_id"),
)


def get_valid_split_size(args, dataset):
    with h5py.File(dataset, "r") as h5file:
        dataset_size = h5file[args.split]["input_ids"].shape[0]
        valid_split_size = min(args.eval_split_limit, dataset_size - max(args.finetune_split_sizes))
    return valid_split_size 
          

def get_counts(df, col_name="counts"):
    ignore = {"table_id", "table_name", "path", "sample_id", "row_id", "col_id"}
    ignore_suffix = {"tn", "tp", "fn", "fp"}
    result_counts = list()
    for agg_level in AGG_LEVELS:
        if not all(k in df.columns for k in agg_level):
            continue
        interesting_cols = list(set(df.columns) - ignore)
        interesting_cols = [c for c in interesting_cols if c.split("_")[-1] not in ignore_suffix]
        for col in interesting_cols:
            counts = _get_counts(agg_level, df, col, col_name=col_name)
            result_counts.append((agg_level, counts))
    return result_counts


def _get_counts(agg_level, df, col, agg_func="sum", col_name="counts"):
    df = df.copy()
    if len(df) == 0:
        return pd.DataFrame(columns=[col_name], index=pd.Index([], name=col))
    categorical = isinstance(df[col].iloc[0], str) or col == "db_operator"
    groupby = list(agg_level) if not categorical else [*agg_level, col]
    aggregated = df.groupby(by=groupby).agg("min" if categorical else {col: agg_func}).reset_index()
    aggregated[col_name] = 1
    x = aggregated.groupby(by=col).agg({col_name: "sum"}).reset_index()
    x[col_name] = x[col_name].astype(int)
    return x


def get_preprocessed_dataset(dataset):
    if dataset.suffix == ".h5":
        return dataset
    log_file = dataset / "preprocessing.log"
    paths = {}
    last_start_time = None
    with open(log_file) as f:
        for line in f:
            if line.startswith("STARTED"):
                start_time = line[8:27]
                output_path = next(f)[14:].strip()
                paths[start_time] = output_path
                next(f)
            elif line.startswith("FINISHED"):
                last_start_time = line[9:28]
    return Path(paths[last_start_time])


def get_model_name_or_path(model_dir):
    log_file = model_dir / "training.log"
    paths = {}
    last_start_time = None
    with open(log_file) as f:
        for line in f:
            if line.startswith("STARTED"):
                start_time = line[8:27]
                next(f)
                target_path = next(f)[14:].strip()
                paths[start_time] = target_path
                next(f)
            elif line.startswith("FINISHED"):
                last_start_time = line[9:28]
    return Path(paths[last_start_time])

def logging_begin(model, train_dataset=None, valid_dataset=None):
    """Log dataset and model characteristics."""
    logger.info(f"Using pre-trained model: {model}")
    logger.info("Number of trainable parameters: %s",
                str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if train_dataset is not None and hasattr(train_dataset, "__len__"):
        logger.info(f"Training dataset size: {len(train_dataset)}")
    if valid_dataset is not None and hasattr(valid_dataset, "__len__"):
        logger.info(f"Development dataset size: {len(valid_dataset)}")


def get_target_path(args, name="pretrained", dataset=None, method=None, exist_ok=False):
    """Get the path to store final weights."""
    target_base_path = Path((args.model_dir) / name)
    if dataset is not None and method is not None:
        target_base_path = Path((args.model_dir) / dataset / method / name)
    basename = get_basename(args.local_rank, args)
    target_path = target_base_path / basename
    target_path.mkdir(parents=True, exist_ok=exist_ok)
    return target_path

def get_target_path_finetuning(args, dataset, method, exist_ok=False):
    """Get the path to store final weights."""
    if args.store:
        return get_target_path(args, "finetuned", dataset, method, exist_ok=exist_ok)


def get_checkpoint_path(args, dataset=None, method=None):
    """Get checkpoint directory."""
    if hasattr(args, "resume") and args.resume:
        logger.info(f"Resume run of checkpoint {args.resume}.")
        return args.resume
    
    if args.save_steps >= 100000000:
        return None

    checkpoint_base_path = Path((args.model_dir) / "checkpoint")
    if dataset is not None and method is not None:
        checkpoint_base_path = Path((args.model_dir) / dataset / method / "checkpoint")
    basename = get_basename(args.local_rank, args)
    checkpoint_path = checkpoint_base_path / basename
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def get_basename(local_rank, args):
    """Get the basename of the checkpoint path."""
    git_hash, git_msg, args_hash = get_git_arg_hashes(args, return_git_msg=True)
    basename = "_".join((git_hash, git_msg)) + "/" + "_".join((get_date_prefix(), str(local_rank), args_hash)) 
    return basename


def get_logging_path(args, dataset=None, method=None):
    """Choose a unique logging path."""
    base_logging_dir = Path(args.model_dir / "logging")
    if dataset is not None and method is not None:
        base_logging_dir = Path(args.model_dir / dataset / method / "logging")
    basename = get_basename(args.local_rank, args)
    logging_dir = base_logging_dir / basename
    logging_dir.mkdir(parents=True, exist_ok=True)

    pip_freeze = subprocess.run(["pip", "freeze"], capture_output=True, check=True).stdout.decode("utf-8")
    git_status = subprocess.run(["git", "status"], capture_output=True, check=True).stdout.decode("utf-8")
    git_diff = subprocess.run(["git", "diff"], capture_output=True, check=True).stdout.decode("utf-8")
    os = subprocess.run(["uname", "-a"], capture_output=True, check=True).stdout.decode("utf-8")

    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, check=True).stdout.decode("utf-8")
        nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, check=True).stdout.decode("utf-8")
    except (FileNotFoundError, subprocess.CalledProcessError):
        nvcc, nvidia_smi = ("Not available", ) * 2

    with open(logging_dir / "env-info.txt", "w") as f:
        print("ARGS:", file=f)
        for arg in vars(args):
            print(arg, getattr(args, arg) or "", file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("CUDA AVAILABLE:", file=f)
        print(torch.cuda.is_available(), file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("PYTHON VERSION:", file=f)
        print(sys.version, file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("PIP FREEZE:", file=f)
        print(pip_freeze, file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("GIT STATUS:", file=f)
        print(git_status, file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("GIT DIFF:", file=f)
        print(git_diff, file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("OS:", file=f)
        print(os, file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("NVCC:", file=f)
        print(nvcc, file=f)
        print("\n", "*" * 300, "\n", file=f)
        print("NVIDIA-SMI:", file=f)
        print(nvidia_smi, file=f)
        print("\n", "*" * 300, "\n", file=f)
    return logging_dir


def table_vec_to_dict(table_vec, tokenizer, has_synonyms=False):
    row_start = 0
    data, row_id_tokens = [], []
    for row_end in np.where(table_vec == -1)[0]:
        if row_start == row_end:
            break

        first_sep = np.where(table_vec[row_start: row_end] == 102)[0]  # get tokens of 1st cell
        first_sep = first_sep[0] + row_start if len(first_sep) > 0 else row_end
        row_id_tokens.append(table_vec[row_start: first_sep])

        row = tokenizer.decode(table_vec[row_start: row_end])
        row_data = dict()
        for cell in filter(bool, map(str.strip, row.split("[SEP]"))):
            col_name, values = map(str.strip, cell.split("|"))
            values = map(str.strip, values.split("-, -"))
            if has_synonyms:
                values = (set(map(str.strip, v.split("- / -"))) for v in values)
            row_data[col_name] = list(values)

        row_start = row_end + 1
        data.append(row_data)
    return data, row_id_tokens

def compute_tp_fp_fn(data_pred, data_label, matching_rows):
    tp, data_pred, data_label = count_tp(matching_rows, data_pred, data_label)
    fp = count_leftover(data_pred)
    fn = count_leftover(data_label)
    return tp, fp, fn

def count_leftover(data):
    result = 0
    for row in data:
        for col in row:
            result += len(row[col])
    return result

def count_tp(matching_rows, data_pred, data_label):
    data_pred, data_label = deepcopy(data_pred), deepcopy(data_label)
    tp = 0
    for i, j in matching_rows:  # Corresponding rows
        row_pred = data_pred[i]
        row_label = data_label[j]
        for col in set(row_pred) & set(row_label):  # Corresponding columns
            cell_pred = row_pred[col]
            cell_label = row_label[col]

            matched_pred = set()
            matched_label = set()
            for a, pred_value in enumerate(cell_pred):  # Check for matches
                for b, label_value in enumerate(cell_label):
                    if b in matched_label:
                        continue
                    if pred_value in label_value:
                        matched_pred.add(a)
                        matched_label.add(b)
                        break

            # remove true positives
            row_pred[col] = [pred_value for a, pred_value in enumerate(cell_pred) if a not in matched_pred]
            row_label[col] = [label_value for b, label_value in enumerate(cell_label) if b not in matched_label]
            tp += len(matched_pred)
    return tp, data_pred, data_label

def match_rows(pred_row_id_tokens, label_row_id_tokens):
    matching_tokens = np.zeros((len(pred_row_id_tokens), len(label_row_id_tokens)))
    for i, pred_tokens in enumerate(pred_row_id_tokens):
        for j, label_tokens in enumerate(label_row_id_tokens):
            matching_tokens[i, j] = len(set(pred_tokens.tolist()) & set(label_tokens.tolist()))
    result = []
    while len(result) < min(matching_tokens.shape):
        argmax = matching_tokens.argmax()
        i = int(argmax / matching_tokens.shape[1])
        j = int(argmax % matching_tokens.shape[1])
        result.append((i, j))
        matching_tokens[i, :] = 0
        matching_tokens[:, j] = 0
    return result