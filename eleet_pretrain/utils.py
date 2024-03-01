"""General utilities."""

import argparse
import sys
import logging
from pathlib import Path
from itertools import chain, zip_longest
import hashlib
import subprocess
import numpy as np
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


class DebugUnderlining():
    def __init__(self, name, abbreviation, start, end, col_ids, normalized=None, no_answers=None):
        self.name = name
        self.abbreviation = abbreviation
        self.start = start
        self.end = end
        self.col_ids = col_ids
        self.normalized = normalized
        self.no_answers = no_answers

def table_from_dataframe(identifier, df, nlp_model):
    """Transform a Dataframe to a TaBERT table."""
    # Copied from TaBERT.preprocess.table.parse
    from table_bert.table import Table, Column
    from preprocess import data_utils

    columns = []
    sampled_values = []
    for col_ids, col_name in enumerate(df.columns):
        sample_value = None
        for _, row in df.iterrows():
            cell_val = row[col_ids] or ""
            if len(cell_val.strip()) > 0:
                sample_value = cell_val
                break

        sampled_values.append(sample_value or "")

    parsed_values = nlp_model.pipe(sampled_values)
    for col_id, sampled_value_annot in enumerate(parsed_values):
        tokenized_value = [token.text for token in sampled_value_annot]
        ner_tags = [token.ent_type_ for token in sampled_value_annot]
        # pos_tags = [token.pos_ for token in sampled_value_annot]

        sample_value_entry = {
            'value': sampled_value_annot.text,
            'tokens': tokenized_value,
            'ner_tags': ner_tags
        }

        col_name = df.columns[col_id]
        col_type = data_utils.infer_column_type_from_sampled_value(sample_value_entry)

        columns.append(Column(col_name, col_type, sample_value=sample_value_entry))

    t = Table(id=identifier, header=columns, data=df)
    return t

def visualize_single(tokenizer, input_ids, token_type_ids, sequence_mask, is_training=None,
                     masked_context_token_labels=None, pred_mlm=None, underlinings=(),
                     print_func=print):
    """Visualize a single example."""
    # content, evidence and labels
    if is_training is not None:
        print_func("TRAINING DEBUG" if is_training else "EVALUATION DEBUG")
    evidences = []
    contexts = []
    for i in range(len(input_ids)):
        context, evidence = tuple(
            tokenizer.decode(input_ids[i][(token_type_ids[i] == j) & sequence_mask[i].bool()]) for j in (0, 1)
        )
        if evidence:
            evidences.append(evidence)
        if context:
            contexts.append(context)
            padding = max([len(u.name) for u in underlinings] + [14]) + 1  # +1 bc colon
            print_func(f"{f'Context {i + 1}:':<{padding}} {context}")
            if masked_context_token_labels is not None:
                print_func(f"{'Masked Text:':<{padding}} {masked_text(tokenizer, input_ids[i], masked_context_token_labels[i])}")
            if pred_mlm is not None:
                print_func(f"{'Predicted Text:':<{padding}} {masked_text(tokenizer, input_ids[i], pred_mlm[i])}")

            for u in underlinings:
                c = u.col_ids[i] if u.col_ids is not None else None
                ul = underlining(tokenizer, input_ids[i], u.start[i], u.end[i], c, f'{u.abbreviation}{i + 1}.')
                print_func(f"{f'{u.name}:':<{padding}} {ul}")
            
            # for u in underlinings:
            #     if u.normalized is not None:
            #         print_func(f"{u.name} Normalized: " + normalized_answers_str(tokenizer, u.normalized[i], u.col_ids[i],
            #                                                                      f"{u.abbreviation}{i + 1}."))
    if contexts:
        for u in underlinings:
            if u.no_answers is not None:
                no_answers_str = ", ".join(f"{u.abbreviation}{r + 1}.{c}"
                                           for r, c in zip(u.no_answers[0].tolist(), u.no_answers[1].tolist()))
                print_func(f"{u.name} No answers: {no_answers_str}")
    print_func(table(print_func, evidences))


def debug_transform_binary(input_ids, labels, query_coords):
    labels = to_iob(labels)
    return debug_transform(input_ids, labels, query_coords)


def to_iob(labels):
    orig_shape = labels.shape
    if len(orig_shape) == 3:
        labels = labels.reshape((-1, orig_shape[-1]))
    start_i, start_j = torch.where(labels[:, 1:] & ~labels[:, :-1])
    labels = labels.float()
    labels[start_i, start_j + 1] = 2
    labels = labels.reshape(orig_shape)
    return labels


def debug_transform(input_ids, labels, query_coords):
    q_id, t_id = torch.where(labels == 2)
    if query_coords is None:
        r_id = torch.arange(input_ids.size(0), device=input_ids.device)
        c_id = torch.zeros(input_ids.size(0), device=input_ids.device)
    else:
        r_id, c_id = query_coords[q_id, :2].T

    answer_start, answer_end, answer_col_ids = debug_get_answer_spans(
            input_ids, labels, q_id, t_id, r_id, c_id)

    return answer_start, answer_end, answer_col_ids


def get_no_answers(labels, query_coords):
    missing_r_id, missing_c_id = query_coords[(labels.sum(1) == 0) & (query_coords[:, 0] >= 0), :2].T
    return missing_r_id, missing_c_id


def debug_get_answer_spans(input_ids, labels, q_id, t_id, r_id, c_id):  # replace with get_answer_end TODO
    answer_start = [[] for _ in range(input_ids.size(0))]
    answer_end = [[] for _ in range(input_ids.size(0))]
    answer_col_ids = [[] for _ in range(input_ids.size(0))]
    for q, t, r, c in zip(q_id, t_id, r_id, c_id):
        answer_start[r].append(t.item())
        answer_col_ids[r].append(c.item())
        answer_end[r].append(next(x for x in range(t + 1, input_ids.size(1) + 1)
                                    if x == input_ids.size(1) or labels[q, x] != 1))
    max_answers = max(map(len, answer_end))
    answer_start = torch.tensor([x[:max_answers] + [0] * max(0, max_answers - len(x)) for x in answer_start],
                                device=input_ids.device)
    answer_col_ids = torch.tensor([x[:max_answers] + [0] * max(0, max_answers - len(x)) for x in answer_col_ids],
                                    device=input_ids.device)
    answer_end = torch.tensor([x[:max_answers] + [0] * max(0, max_answers - len(x)) for x in answer_end],
                                device=input_ids.device)
    return answer_start, answer_end, answer_col_ids


def normalized_answers_str(tokenizer, normalized_answers, answer_col_ids, prefix):
    mask = normalized_answers[:, 0] > 0
    result = ""
    prev_col_id = -1
    already_printed = set()
    for n, col_id in sorted(zip(normalized_answers[mask], answer_col_ids[mask]), key=lambda x: x[1]):
        col_id = col_id.item()
        pre = ""
        if col_id != prev_col_id:
            pre = f"{prefix}{col_id}: "
            already_printed = set()
        decoded = tokenizer.decode(n[n > 102])
        if decoded not in already_printed:
            result += f"{pre}{decoded}, "
        prev_col_id = col_id
        already_printed.add(decoded)
    return result
    # for i, (token_ids, starts, ends, col_ids, normeds) in enumerate(zip(input_ids, answer_start, answer_end,
    #                                                                     answer_col_ids, normalized_answers)):
    #     print_func(underlining(tokenizer, token_ids, starts, ends))


def table(print_func, evidences):
    result = "\n"
    evidence_splitted = [[[v.strip(" ,") for v in c.split("|")] for c in e.split("[SEP]")[:-1]] for e in evidences]
    max_evidence_len = max([len(e) for e in evidence_splitted])
    longest_evidence = next(x for x in evidence_splitted if len(x) == max_evidence_len)
    values = [[cell[2] if cell is not None else None for cell in col] for col in zip_longest(*evidence_splitted)]
    header = [" | ".join(cell[0:2]) for cell in longest_evidence]
    lengths = [max(len(h), *map(lambda x: len(x) if x is not None else 0, vs)) for h, vs in  zip(header, values)]
    for i, row in enumerate(chain((header, ), zip(*values))):
        for j, (c, l) in enumerate(zip(row, lengths)):
            if c is None:
                continue
            result += f"{i} "
            result += c.center(l)
            result += f" {j} [SEP] "
        result += (row[-1] if len(row) > len(lengths) else "") + "\n"
    result += "\n"
    return result
    


    # mention = evidence_mentions_underlining(tokenizer, evidence_mentions, input_ids)
    # if mention:
    #     print_func(f"Mention:      {mention}")

    # for i in range(num_answers):
    #     if labels is not None:
    #         print_func(f"{i}. Labels:    {label_underlining(tokenizer, labels[i], input_ids[0])}")

    #     if predicted_labels is not None:
    #         print_func(f"{i}. Predicted: {label_underlining(tokenizer, predicted_labels[i], input_ids[0])}")

    #     if labels is not None:
    #         answers = compute_answers(tokenizer=tokenizer, labels=labels[i], input_ids=input_ids[0])
    #         print_func(f"{i}. True answer(s): {answers}")

    #     if normalized_answers is not None:
    #         mask = normalized_answers[i][:, 0] != 0
    #         norm = set(tokenizer.decode(x) for x in normalized_answers[i][mask])
    #         norm = set(x.replace("[CLS]", "").replace("[PAD]", "").replace("[SEP]", "").strip() for x in norm)
    #         print_func(f"{i}. Normalized answer(s): {norm}")

    #     # predictions
    #     if predicted_labels is not None:
    #         pred_answers = compute_answers(
    #             tokenizer=tokenizer,
    #             labels=predicted_labels[i],
    #             input_ids=input_ids[0]
    #         )
    #         print_func(f"{i}. Predicted answer(s): {pred_answers}")
    #     if predicted_normalized_answers is not None:
    #         mask = predicted_normalized_answers[i][:, 0] != 0
    #         norm = set(tokenizer.decode(x) for x in predicted_normalized_answers[i][mask])
    #         norm = set(x.replace("[CLS]", "").replace("[PAD]", "").replace("[SEP]", "").strip() for x in norm)
    #         print_func(f"{i}. Predicted normalized answer(s): {norm}")
    #     print_func("\n")


def compute_answers(tokenizer, labels, input_ids):
    """Compute the answers from labels and tokens."""
    answer_labels = labels[labels > 0]
    answer_tokens = input_ids[labels > 0]
    answers = list()
    current = []
    for label, token in zip(answer_labels, answer_tokens):
        if label == 2 and current:
            answers.append(tokenizer.decode(current))
            current = []
        current.append(token)
    if current:
        answers.append(tokenizer.decode(current))
    return answers

def underlining(tokenizer, input_ids, starts, ends, col_ids, prefix):
    """Underline the text to visualize which tokens have which label."""
    result = ""
    correct_shift = 0
    prev = 0
    skip_adding_space = True
    starts = starts.reshape(-1, )
    ends = ends.reshape(-1, )
    mask = starts > 0
    col_ids_iter = col_ids[mask] if col_ids is not None else [None] * len(mask)
    for start, end, col_id in sorted(zip(starts[mask], ends[mask], col_ids_iter)):
        if start < prev:
            continue
        pre = tokenizer.decode(input_ids[prev:start])
        this = tokenizer.decode(input_ids[start:end])
        this_len, pre_len = len(this), len(pre)

        if this.startswith("##"):
            this_len -= 2

        if pre.startswith("##"):
            pre_len -= 2

        if not any(pre.startswith(c) for c in ("##", ",", ".", "'")) and not skip_adding_space:
            result += " "
        result += " " * max(0, pre_len + correct_shift)
        skip_adding_space = pre.endswith("'")
        if not any(this.startswith(c) for c in ("##", ",", ".", "'")) and not skip_adding_space:
            result += " "
        col_id = str(col_id.item()) if col_id is not None else ""
        this_underline = (prefix + col_id).center(this_len, "-")
        result += this_underline
        skip_adding_space = this.endswith("'")
        correct_shift = this_len - len(this_underline) - max(0, -(pre_len + correct_shift))
        prev = end
    return result

def label_underlining(tokenizer, labels, input_ids, chars=" IS"):
    """Underline the text to visualize which tokens have which label."""
    result = ""
    skip_adding_space = True
    if not isinstance(labels, tuple):
        labels = (labels, )
        chars = (chars, )
    for i, input_id in enumerate(input_ids):
        decoded = tokenizer.decode(input_id)
        length = len(decoded)
        if not decoded.startswith("##") and decoded not in (",", ".", "'") and not skip_adding_space:
            result += " "

        if decoded.startswith("##"):
            length -= 2
        
        for j in range(length):
            result += chars[j % len(labels)][labels[j % len(labels)][i]]
        skip_adding_space = decoded in "'"
    return result


def evidence_mentions_underlining(tokenizer, evidence_mentions, input_ids):
    if evidence_mentions is None:
        return None
    num_cols = evidence_mentions.shape[1]
    evidence_mentions = evidence_mentions.reshape(-1, evidence_mentions.shape[-1]).int()
    evidence_mentions *= np.arange(1, evidence_mentions.shape[0] + 1).reshape(-1, 1)
    evidence_mentions[evidence_mentions == 0] = evidence_mentions.max()
    evidence_mentions = evidence_mentions.min(0)[0]
    evidence_mentions[evidence_mentions == evidence_mentions.max()] = 0
    em_pair = (
        (evidence_mentions / num_cols).ceil().int(),
        (evidence_mentions % num_cols)
    )

    return label_underlining(tokenizer, em_pair, input_ids[0],
                             chars=(" 12345", " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def blocks(files, size=65536):
    """Iterate over blocks of the file."""
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(file):
    """Count the lines in a file."""
    with open(file, "r", encoding="utf-8", errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f)) + 1


def logging_setup(log_level, log_file=None):
    """Set up logging."""
    if log_file:
        log_file.parent.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(format="%(levelname)s %(asctime)s %(name)s: %(message)s",
                            filemode="w",
                            filename=log_file,
                            level=log_level)
    else:
        logging.basicConfig(format="%(levelname)s %(asctime)s %(name)s: %(message)s",
                            level=log_level)


try:
    _git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, check=True) \
        .stdout.decode("utf-8").strip()
except subprocess.CalledProcessError:
    _git_hash = "0"

try:
    _git_msg = subprocess.run(["git", "log", "-1", "--pretty=%B"], capture_output=True, check=True) \
        .stdout.decode("utf-8").replace(" ", "-").strip()
except subprocess.CalledProcessError:
    _git_msg = "0"


def get_git_arg_hashes(args, return_git_msg=False):
    """Get hashes to avoid overriding when executing a new load data / eval / training run."""
    m = hashlib.sha256()
    m.update("-".join(["-".join(map(str, (k, v))) for k, v in sorted(vars(args).items())
                       if isinstance(v, (Path, str, int, float, list, tuple))]).encode("utf-8"))
    args_hash = m.hexdigest()[:8]
    if not return_git_msg:
        return _git_hash, args_hash
    return _git_hash, _git_msg, args_hash


def get_date_prefix():
    from eleet_pretrain import _program_start
    date_prefix = _program_start.strftime("%Y-%m-%d_%H-%M-%S")
    return date_prefix

def insert_into_global_log_begin(file_path, logging_path, checkpoint_path=None, output_path=None, target_path=None):
    """Insert message into the training logs."""
    with open(file_path, "a") as f:
        date = datetime.now()
        date_str = date.strftime("%d.%m.%Y %H:%M:%S")
        print("STARTED", date_str, " ".join(sys.argv), file=f)
        if checkpoint_path:
            print("\tCheckpoint path:", Path(checkpoint_path).absolute(), file=f)
        if output_path:
            print("\tOutput path:", Path(output_path).absolute(), file=f)
        if target_path:
            print("\tTarget path:", Path(target_path).absolute(), file=f)
        print("\tLogging path:", Path(logging_path).absolute(), file=f)
    return date


def insert_into_global_log_end(file_path, start_date, exception=None):
    """Insert message into the training logs."""
    msg = "FAILED" if exception is not None else "FINISHED"
    with open(file_path, "a") as f:
        date = datetime.now()
        date_str = date.strftime("%d.%m.%Y %H:%M:%S")
        start_str = start_date.strftime("%d.%m.%Y %H:%M:%S")
        print(msg, start_str, "until", date_str, "(", date - start_date, ")", " ".join(sys.argv), file=f)
        if exception is not None:
            print("\tException:", type(exception).__name__, exception, file=f)
            logger.error("Unhandled exception occurred. Cannot recover.", exc_info=True)


def col_id_underlining(tokenizer, column_ids, input_ids, token_type_ids):
    return label_underlining(tokenizer, column_ids[token_type_ids == 1], input_ids[token_type_ids == 1],
                             chars="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

def masked_text(tokenizer, input_ids, masked_context_token_labels):
    positions = (masked_context_token_labels >= 0) * torch.arange(len(masked_context_token_labels),
                                                                  device=masked_context_token_labels.device)
    positions = positions[masked_context_token_labels >= 0]
    masked_context_token_labels = masked_context_token_labels[masked_context_token_labels >= 0]
    result = ""
    for p, label in zip(positions, masked_context_token_labels):
        prefix_len = len(tokenizer.decode(input_ids[:p])) + 1
        if prefix_len > len(result):
            result += " " * (prefix_len - len(result))
        else:
            result = result[:prefix_len]
        result += tokenizer.decode([label])
    return result


def compute_span_distances(embeddings, duplicate_detect_layer, duplicate_detect_threshold):
    selector = torch.stack((torch.arange(embeddings.size(0)).repeat_interleave(embeddings.size(0)),
                            torch.arange(embeddings.size(0)).repeat(embeddings.size(0))))
    em_0, em_1 = (embeddings[selector[0]].float(), embeddings[selector[1]].float())
    logits = compute_span_similarities(duplicate_detect_layer, duplicate_detect_threshold, em_0, em_1)
    return -logits.view(embeddings.size(0), embeddings.size(0))

def compute_span_similarities(duplicate_detect_layer, duplicate_detect_threshold, em_0, em_1):
    em_0 = duplicate_detect_layer(em_0)
    logits = (em_0 * em_1).sum(1) # / (torch.norm(em_0, p=2, dim=1) * torch.norm(em_1, p=2, dim=1))
    logits = logits - duplicate_detect_threshold
    return logits


class DummyArgumentParser():
    def __init__(self):
        self.arguments = dict()
        self.values = dict()
    
    def to_var_name(self, key):
        return key.lstrip("-").replace("-", "_")

    def add_argument(self, key, *args, **kwargs):
        self.arguments[key] = (args, kwargs)

    def set_value(self, key, value):
        del self.arguments[key]
        self.values[key] = value

    def parse_args(self):
        parser = argparse.ArgumentParser(conflict_handler="resolve")
        for key, (args, kwargs) in self.arguments.items():
            parser.add_argument(key, *args, **kwargs)
        args = parser.parse_args()
        for key, value in self.values.items():
            setattr(args, self.to_var_name(key), value)
        return args


def rm_conflicting_answers(answers):
    if len(answers) == 0:
        return answers
    answers = answers.groupby(by=["row_id", "col_id", "query_id", "answer_start"]).apply(
        lambda x: x[x["answer_end"] == max(x["answer_end"])].iloc[[0]] \
            .reset_index("answer_id").reset_index(drop=True)
    )
    answers.reset_index(["answer_start", None], drop=True, inplace=True)
    answers.set_index("answer_id", append=True, inplace=True)
    return answers


def kwargs_dict(key_choices, value_type=str, value_obligatory=True):
    class kwargs_action(argparse.Action):
        """
        argparse action to split an argument into KEY=VALUE form
        on append to a dictionary.
        """

        def __call__(self, parser, args, values, option_string=None):
            try:
                d = {y[0]: value_type(y[1]) if len(y) > 1 or value_obligatory else None
                    for y in (x.split("=") for x in values)}
                if any(x not in key_choices for x in d):
                    raise argparse.ArgumentError(self, f"Choices are {key_choices}")
            except ValueError as ex:
                raise argparse.ArgumentError(self, f"Could not parse argument \"{values}\" as k1=v1 k2=v2 ... format")
            setattr(args, self.dest, d)
    return kwargs_action
