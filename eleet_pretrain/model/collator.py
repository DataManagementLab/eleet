import itertools
import logging
import torch
import numpy as np
from transformers import DefaultDataCollator

logger = logging.getLogger(__name__)


class PRETRAINING_OBJECTIVE():
    RELEVANT_TEXT_DETECTION = 1  # align rows w/ text
    HEADER_QUERIES = 2  # align columns w/ text
    CUTOUT_COLUMN_NAMES = 3
    DEDUPLICATION = 4
    MASKED_LANGUAGE_MODEL = 5
    SHUFFLE_COLUMNS = 6
    SHUFFLE_ROWS = 7


class EleetDataCollator(DefaultDataCollator):

    def __init__(self, config, tokenizer, disabled=(), model=None, is_pretraining=False, is_finetuning=False,
                 seed=42):
        self.config = config
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.is_pretraining = is_pretraining
        self.is_finetuning = is_finetuning
        self.tokenizer = tokenizer
        self.np_rng = np.random.default_rng(seed + 1)
        self.vocab_list = self.tokenizer.vocab.values()
        self.model = model
        self.disabled = disabled
        self.is_complex_operation = False
        self.add_ground_truth = False
        super().__init__()

    return_tensors: str = "pt"

    def setup(self, collated):
        collated["query_labels"] = None
        collated["query_coords"] = None

    def cleanup(self, collated):
        for x in ("header_query_end", "header_query_start", "header_query_col_ids", "row_label_span", "answer_end",
                  "answer_start", "answer_col_ids", "relevant_text", "answer_dep_a_start", "answer_dep_qid",
                  "answer_qid", "deduplication_labels", "query_normed", "mask_token_positions", "mention_col_ids"):
            if x in collated:
                del collated[x]

    def __call__(self, features, return_tensors=None):
        collated = super().__call__(features=features, return_tensors=return_tensors)
        batch_size = len(collated["input_ids"])
        num_rows = (collated["input_ids"][:, :, 0] != 0).sum(axis=1)
        max_num_rows = collated["input_ids"].size(1)

        # align_labels, align_coords = None, None
        self.setup(collated)
        if self.is_pretraining:
            if PRETRAINING_OBJECTIVE.SHUFFLE_ROWS not in self.disabled:
                collated = self.shuffle_rows(collated, batch_size, max_num_rows, num_rows)
            if PRETRAINING_OBJECTIVE.SHUFFLE_COLUMNS not in self.disabled:
                collated = self.shuffle_columns(collated, batch_size, num_rows)
            if PRETRAINING_OBJECTIVE.MASKED_LANGUAGE_MODEL not in self.disabled:
                collated = self.mlm(collated, batch_size, num_rows)
            if PRETRAINING_OBJECTIVE.RELEVANT_TEXT_DETECTION not in self.disabled:
                collated = self.prepare_relevant_text_labels(collated, batch_size, num_rows)
            if PRETRAINING_OBJECTIVE.HEADER_QUERIES not in self.disabled:
                collated = self.prepare_header_queries(collated, batch_size)
            if PRETRAINING_OBJECTIVE.DEDUPLICATION not in self.disabled:
                collated = self.add_deduplication(collated)
        elif self.is_finetuning:
            collated = self.add_deduplication_finetuning(collated)  # TODO why separate method for finetuning?
            collated = self.prepare_header_queries(collated, batch_size)
        collated = self.prepare_queries(collated, batch_size)

        if self.is_pretraining and self.model is not None and self.model.training \
                and PRETRAINING_OBJECTIVE.CUTOUT_COLUMN_NAMES not in self.disabled:
            # inserts MASK tokens, so must be executed after prep of labels
            collated = self.cutout_col_labels(collated, batch_size, num_rows)

        self.cleanup(collated)
        return collated

    def mlm(self, collated, batch_size, num_rows):
        collated["masked_context_token_labels"] = torch.full_like(collated["input_ids"], -1)
        for b in range(batch_size):
            for r in range(num_rows[b]):
                self.mask_context(collated, b, r)
        return collated

    def mask_context(self, collated, b, r):
        """MLM objects. Sample tokens to masked."""
        num_context_tokens = collated["context_token_mask"][b, r].sum()
        context_token_indices = torch.arange(1, max(1, num_context_tokens))  # minus CLS token
        max_context_token_to_mask = min(self.config.max_predictions_per_seq, len(context_token_indices))
        num_context_tokens_to_mask = min(
            max_context_token_to_mask,
            max(1, int(len(context_token_indices) * self.config.masked_context_prob))
        )
        masked_context_token_indices = []
        if num_context_tokens_to_mask > 0:
            masked_context_token_indices = sorted(self.np_rng.choice(context_token_indices,
                                                                     num_context_tokens_to_mask, replace=False))

        masked_context_token_labels = collated["input_ids"][b, r][masked_context_token_indices]
        self.insert_context_mask_tokens(collated, b, r, masked_context_token_indices, masked_context_token_labels)
        collated["masked_context_token_labels"][b, r, masked_context_token_indices] = masked_context_token_labels

    def insert_context_mask_tokens(self, collated, b, r, masked_context_token_indices, masked_context_token_labels):
        for token_relative_idx, token in enumerate(masked_context_token_labels):
            # 80% of the time, replace with [MASK]
            if self.np_rng.random() < 0.8:
                masked_token = self.tokenizer.mask_token_id
            else:
                # 10% of the time, keep original
                if self.np_rng.random() < 0.5:
                    masked_token = token
                # 10% of the time, replace with random word
                else:
                    masked_token = self.np_rng.choice(list(self.vocab_list))

            token_idx = masked_context_token_indices[token_relative_idx]
            collated["input_ids"][b, r][token_idx] = masked_token

    def shuffle_rows(self, collated, batch_size, max_num_rows, num_rows):
        perms = []
        for b in range(batch_size):
            perm = torch.randperm(num_rows[b], generator=self.rng)
            perm = torch.hstack((perm, torch.arange(num_rows[b], max_num_rows)))
            perms.append(perm + b * max_num_rows)
        perm = torch.hstack(perms)
        shuffled = dict()
        for k, v in collated.items():
            if v is None:
                continue
            if k in ("idx", "table_id"):
                shuffled[k] = v
                continue
            p = v.view(batch_size * max_num_rows, *v.size()[2:])[perm]
            shuffled[k] = p.view(batch_size, max_num_rows, *v.size()[2:])
        return shuffled

    def shuffle_columns(self, collated, batch_size, num_rows):
        token_ids, column_ids = self.get_shuffle_columns_ids(collated, batch_size, num_rows)

        b_id, r_id, t_id, p_id = token_ids
        for k in ("input_ids", "column_token_position_to_column_ids"):
            collated[k][b_id, r_id, t_id] = collated[k][b_id, r_id, p_id]
        b_id, c_id, p_id = column_ids
        for k in ("column_token_position_to_column_ids", "answer_col_ids", "header_query_col_ids"): # "align_col_ids"):
            orig = torch.clone(collated[k])
            for b, c, p in zip(b_id, c_id, p_id):
                mask = orig[b] == p
                collated[k][b][mask] = c
        return collated

    def get_shuffle_columns_ids(self, collated, batch_size, num_rows):
        result_token, result_column = list(), list()
        for b in range(batch_size):
            num_permuted_columns = collated["table_mask"][b][:num_rows[b]].sum(axis=1).min()
            permuted_col_ids = torch.randperm(num_permuted_columns, generator=self.rng)
            col_ids = torch.arange(num_permuted_columns)
            col_batch_ids = torch.full_like(col_ids, b)
            result_column.append((col_batch_ids, col_ids, permuted_col_ids))

            column_token_position_to_column_ids = collated["column_token_position_to_column_ids"][b]
            p_id, r1_id = self.get_permuted_token_id(column_token_position_to_column_ids, num_rows[b], permuted_col_ids)
            t_id, r2_id = self.get_permuted_token_id(column_token_position_to_column_ids, num_rows[b], col_ids)
            b_id = torch.full((p_id.size(0), ), b)
            assert (r1_id == r2_id).all()
            result_token.append((b_id, r1_id, t_id, p_id))
        return tuple(torch.hstack(x) for x in zip(*result_token)), tuple(torch.hstack(x) for x in zip(*result_column))

    def get_permuted_token_id(self, column_token_position_to_column_ids, num_rows, perm):
        p_id = []
        r_id = []
        for r in range(num_rows):
            for p in perm:
                col_token_ids = torch.where(column_token_position_to_column_ids[r] == p)[0]
                p_id.append(col_token_ids)
                p_id.append(col_token_ids[-1] + 1)
                r_id.append(torch.tensor([r] * (len(col_token_ids) + 1)))
        return torch.hstack(p_id), torch.hstack(r_id)

    def prepare_queries(self, collated, batch_size):
        labels, coords, normed = self._prepare_answers(collated=collated, batch_size=batch_size)
        labels, coords, normed = self.combine_labels(batch_size, labels, coords, normed, coords_width=5)
        collated["query_labels"] = labels
        collated["query_coords"] = coords
        collated["query_normed"] = normed
        return collated

    def prepare_header_queries(self, collated, batch_size):
        labels, coords = self.hq_to_coords(collated=collated, batch_size=batch_size,
                                         col_ids_name="header_query_col_ids", start_name="header_query_start",
                                         end_name="header_query_end")
        labels, coords = self.combine_labels(batch_size, labels, coords, coords_width=2)
        collated["header_query_labels"] = labels
        collated["header_query_coords"] = coords
        return collated


    def hq_to_coords(self, collated, batch_size, col_ids_name, start_name, end_name):
        if col_ids_name not in collated:
            return None, None

        e_id, r_id, a_id = torch.where(collated[col_ids_name] >= 0)
        c_id = collated[col_ids_name][e_id, r_id, a_id]
        starts = collated[start_name][e_id, r_id, a_id]
        ends = collated[end_name][e_id, r_id, a_id]

        labels = [[] for _ in range(batch_size)]
        coords = [[] for _ in range(batch_size)]
        done = [{} for _ in range(batch_size)]

        for i, r, c, s, e in zip(e_id, r_id, c_id, starts, ends):
            r, c = r.item(), c.item()
            if (r, c) not in done[i]:
                done[i][r, c] = len(done[i])
                labels[i].append(torch.zeros(self.config.max_sequence_len, dtype=int))
                coords[i].append(torch.tensor([r, c], dtype=int))

            j = done[i][r, c]
            labels[i][j][s + 1: e] = 1
            if s > -1 and not labels[i][j][s]:
                labels[i][j][s] = 2
        labels = [torch.vstack(e) if len(e) else None for e in labels]
        coords = [torch.vstack(e) if len(e) else None for e in coords]
        return labels, coords

    def _prepare_answers(self, collated, batch_size):
        e_id, r_id, t_id = torch.where((collated["input_ids"] == 103) & (collated["token_type_ids"] == 1))
        c_id = collated["column_token_position_to_column_ids"][e_id, r_id, t_id]
        a_col_id = collated["answer_col_ids"][e_id, r_id]
        starts = collated["answer_start"][e_id, r_id]
        ends = collated["answer_end"][e_id, r_id]
        normed = collated["normalized_answers"][e_id, r_id]
        qid = collated["answer_qid"][e_id, r_id]
        dep_qid = collated["answer_dep_qid"][e_id, r_id]
        dep_a_start = collated["answer_dep_a_start"][e_id, r_id]

        labels_given = (collated["answer_qid"] > -1).any()

        # aligns answers to query. num_queries x max_num_answers. Indicates which answers corresponds to which query.
        query_answer_alignment = (a_col_id == c_id.view(-1, 1)) & ends.bool()

        query_labels = [[] for _ in range(batch_size)]
        query_coords = [[] for _ in range(batch_size)]
        query_normed = [[] for _ in range(batch_size)]
        label_idx = [{} for _ in range(batch_size)]  # single label tensor for each query

        # iterate over queries
        max_qid = qid.max() if qid.numel() else 0
        backup_qids = np.arange(max_qid + 1, max_qid + 1 + e_id.shape[0])  # in case there are no answers
        for i, r, c, align, starts_row, ends_row, normed_row, qid_row, dep_qid_row, dep_a_start_row, bkp_qid in zip(
           e_id, r_id, c_id, query_answer_alignment, starts, ends, normed, qid, dep_qid, dep_a_start, backup_qids
        ):
            r, c = r.item(), c.item()
            self.init_labels(query_labels[i], query_coords[i], query_normed[i], label_idx[i], r, c,
                             qid_row[align], dep_qid_row[align], dep_a_start_row[align], bkp_qid if labels_given else c)
            for s, e, n, q in zip(starts_row[align], ends_row[align], normed_row[align], qid_row[align]):
                q = q.item()
                j = label_idx[i][r, c, q]
                query_labels[i][j][s + 1: e] = 1
                if not query_labels[i][j][s]:
                    query_labels[i][j][s] = 2
                    query_normed[i][j][s] = n

        query_labels = [torch.vstack(e) if len(e) else None for e in query_labels]
        query_coords = [torch.vstack(e) if len(e) else None for e in query_coords]
        query_normed = [torch.stack(e) if len(e) else None for e in query_normed]
        return query_labels, query_coords, query_normed

    def init_labels(self, query_labels, query_coords, query_normed, label_idx, row_id, col_id,
                    qid_row, dep_qid_row, dep_a_start_row, bkp_qid):
        # multiple labels for the query possible, if it depends on some other query
        added_labels = False
        for q, q_dep, q_dep_start in zip(qid_row, dep_qid_row, dep_a_start_row):
            self.init_query_labels_coords(query_labels, query_coords, query_normed, label_idx,
                                          row_id, col_id, q, q_dep, q_dep_start)
            added_labels = True
        # ensure that each query has at least one set of labels
        if not added_labels:
            self.init_query_labels_coords(query_labels, query_coords, query_normed, label_idx, row_id, col_id, bkp_qid)

    def init_query_labels_coords(self, query_labels, query_coords, query_normed, label_idx, r, c, q,
                                 q_dep=-1, q_dep_start=-1):
        q = q if isinstance(q, int) else q.item()
        if (r, c, q) not in label_idx:
            label_idx[r, c, q] = len(label_idx)
            query_labels.append(torch.zeros(self.config.max_sequence_len, dtype=int))
            query_coords.append(torch.tensor([r, c, q, q_dep, q_dep_start], dtype=int))
            query_normed.append(torch.zeros(self.config.max_sequence_len,
                                            self.config.max_len_answer,dtype=int))

    def combine_labels(self, batch_size, labels, coords, normed=None, coords_width=2):
        # labels, coords = self.concat_labels(labels, coords, batch_size)
        return_normed = normed is not None
        labels_t = torch.zeros(batch_size, self.config.max_num_queries, self.config.max_sequence_len, dtype=int)
        coords_t = torch.full((batch_size, self.config.max_num_queries, coords_width), -1, dtype=int)
        normed_t = torch.zeros((batch_size, self.config.max_num_queries,
                                self.config.max_sequence_len, self.config.max_len_answer), dtype=int)
        normed = normed or [None] * len(labels)
        for i, (l, c, n) in enumerate(zip(labels, coords, normed)):
            if l is None:
                continue
            num_queries = min(len(l), self.config.max_num_queries)
            labels_t[i, :num_queries] = l[:num_queries]
            coords_t[i, :num_queries] = c[:num_queries]
            if n is not None:
                normed_t[i, :num_queries] = n[:num_queries]

            if coords_t[i, :num_queries].shape != torch.unique(coords_t[i, :num_queries], dim=0).shape:
                raise ValueError("Multiple labels for same table cell")

            if len(l) > self.config.max_num_queries:
                logger.warn(f"More queries ({len(l)}) than allowed ({self.config.max_num_queries}). "
                            "Cutting off queries.")

        if not return_normed:
            return labels_t, coords_t
        return labels_t, coords_t, normed_t

    def cutout_col_labels(self, collated, batch_size, num_rows):
        pipe_token_id = self.tokenizer.encode("|")[1]
        non_values = set([self.tokenizer.mask_token_id, self.tokenizer.encode("none")[1], pipe_token_id])
        for i in range(batch_size):
            max_col = collated["column_token_position_to_column_ids"][i].max() + 1
            skip_decision = torch.rand(max_col, generator=self.rng) >= self.config.cutout_col_label_prob
            for c_id, skip in enumerate(skip_decision):
                if skip:
                    continue

                col_mask = collated["column_token_position_to_column_ids"][i] == c_id
                has_value = False
                for j in range(num_rows[i]):
                    col_value = collated["input_ids"][i, j][col_mask[j]]
                    if len(col_value) and col_value[-1].item() not in non_values:
                        has_value = True 
                
                if not has_value:
                    continue

                for j in range(num_rows[i]):
                    if not col_mask[j].any():
                        continue
                    first_pipe = (collated["input_ids"][i, j][col_mask[j]] == pipe_token_id).int().argmax()
                    collated["input_ids"][i, j][torch.where(col_mask[j])[0][:first_pipe]] = self.tokenizer.mask_token_id
        return collated

    def prepare_relevant_text_labels(self, collated, batch_size, num_rows):
        collated["relevant_text_labels"] = torch.zeros_like(collated["input_ids"], dtype=int)
        for i in range(batch_size):
            for j in range(num_rows[i]):
                s, e = collated["relevant_text"][i, j]
                collated["relevant_text_labels"][i, j, s: e] = 1
        return collated

    def add_deduplication(self, collated):
        result = torch.zeros(*collated["input_ids"].shape[:2], *collated["deduplication_labels"].shape[2: 5],
                             collated["input_ids"].shape[-1])
        # iter over every element
        for selector in itertools.product(*map(range, collated["deduplication_labels"].shape[:-1])):
            if (collated["deduplication_labels"][selector] == 0).all():
                continue
            result[(*selector, slice(*collated["deduplication_labels"][selector]))] = 1
            result[(*selector, collated["deduplication_labels"][selector][0])] += 1
            result[(*selector, collated["deduplication_labels"][selector][1] - 1)] += 2
        collated["deduplication"] = result
        return collated

    def add_deduplication_finetuning(self, collated):
        if collated["deduplication_labels"].any():  # TODO remove separate method for finetuning?
            return self.add_deduplication(collated)
        result = torch.zeros(*collated["input_ids"].shape[:2], *collated["deduplication_labels"].shape[2: 5],
                             collated["input_ids"].shape[-1])
        counter = torch.tensor([0, 0])
        # iter over every element
        for b, r, i in zip(*torch.where(collated["answer_start"])):
            qid = collated["answer_qid"][b, r, i]
            normed = collated["normalized_answers"][b, r, i]
            for j, in zip(*torch.where(collated["answer_qid"][b, r] == qid)):
                if j <= i:
                    continue
                normed2 = collated["normalized_answers"][b, r, j]
                t = 1 - (normed == normed2).all().int()
                if counter[t] >= result.shape[3]:
                    continue
                result[b, r, t, counter[t], 0, collated["answer_start"][b, r, i]: collated["answer_end"][b, r, i]] = 1
                result[b, r, t, counter[t], 0, collated["answer_start"][b, r, i]] += 1
                result[b, r, t, counter[t], 0, collated["answer_end"][b, r, i] - 1] += 2
                result[b, r, t, counter[t], 1, collated["answer_start"][b, r, j]: collated["answer_end"][b, r, j]] = 1
                result[b, r, t, counter[t], 1, collated["answer_start"][b, r, j]] += 1
                result[b, r, t, counter[t], 1, collated["answer_end"][b, r, j] - 1] += 2
                counter[t] += 1
        collated["deduplication"] = result
        return collated
