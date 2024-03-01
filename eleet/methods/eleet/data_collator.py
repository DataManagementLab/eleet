from collections import defaultdict
from copy import deepcopy
import logging
from math import ceil
import torch
import numpy as np
from transformers import DefaultDataCollator
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class ELEETFinetuningCollator(DefaultDataCollator):

    def __init__(self, config, tokenizer, seed=42):
        self.config = config
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.tokenizer = tokenizer
        self.np_rng = np.random.default_rng(seed + 1)
        self.num_devices = torch.cuda.device_count()
        self.is_training = True
        super().__init__()

    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        collated = super().__call__(features=features, return_tensors=return_tensors)
        collated = self.add_deduplication_labels(collated)
        return collated

    def add_deduplication_labels(self, collated):
        clustering, num_spans = self.deduplication_clustering(collated)

        i = 0
        batch_size = len(collated["input_ids"])
        result = torch.full((num_spans * 3, 10), -1, dtype=int)
        for col_cluster in clustering.values():
            for span, span_cluster in col_cluster.items():
                perm = torch.randperm(len(span_cluster), generator=self.rng)  # same value
                for j in range(len(span_cluster)):
                    e1, r1, (start1, end1) = span_cluster[perm[j]]
                    e2, r2, (start2, end2) = span_cluster[perm[(j + 1) % len(perm)]]
                    result[i] = torch.tensor([1, e1, r1, start1, end1, e2, r2, start2, end2, batch_size])
                    i += 1
                    if len(span_cluster) > 1:
                        result[i] = torch.tensor([1, e1, r1, start1, end1, e1, r1, start1, end1, batch_size])
                        i += 1

                    if len(col_cluster.keys()) > 1:  # different value
                        other_span = tuple(self.np_rng.choice(list(set(col_cluster.keys()) - {span})))
                        other_span_cluster = col_cluster[other_span]
                        i3 = int(torch.randint(0, len(other_span_cluster), (1,), generator=self.rng)[0])
                        e3, r3, (start3, end3) = other_span_cluster[i3]
                        result[i] = torch.tensor([0, e1, r1, start1, end1, e3, r3, start3, end3, batch_size])
                        i += 1

        max_size = collated["deduplication_labels"].shape[0] * collated["deduplication_labels"].shape[2] * 3
        result = result[result[:, 0] != -1]
        if len(result) > max_size:
            result = result[torch.randperm(len(result), generator=self.rng)[:max_size].sort()[0]]
        else:
            result = F.pad(result, pad=(0, 0, 0, max_size - len(result)), value=-1)
        collated["deduplication_labels"] = result.reshape(collated["input_ids"].shape[0], -1, result.shape[-1])
        return collated

    def deduplication_clustering(self, collated):
        sample_id, row_id, span_id = torch.where(collated["deduplication_labels"][:, :, :, 1] > 0)
        clustering = defaultdict(lambda: defaultdict(list))
        cols = collated["deduplication_labels"][sample_id, row_id, span_id][:, 2: 2 + self.config.max_cell_len].tolist()
        spans_normed = collated["deduplication_labels"][sample_id, row_id, span_id][:, 2 + self.config.max_cell_len:].tolist()
        for e, r, s, col, span in zip(sample_id, row_id, span_id, cols, spans_normed):
            clustering[tuple(col)][tuple(span)] += [(
                int(e), int(r), collated["deduplication_labels"][e, r, s][:2].tolist()
            )]
            
        num_spans = len(spans_normed)
        return clustering, num_spans

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("rng", "np_rng")}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.np_rng = np.random.default_rng(self.seed + 1)
