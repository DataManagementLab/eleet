"""Pytorch dataset for TRex-dataset."""

from contextlib import contextmanager
from pathlib import Path
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import tempfile
from tqdm import tqdm
from sys import stdout
from math import ceil


IGNORE_KEYS = {"col_shuffled_ids", "mention_end", "mention_start", "num_cell_tokens",
               "num_cell_tokens_masked", "num_cells_masked", "num_context_tokens_masked", "origin", "statistics",
               "table_size", "detailed_statistics", "text_ids"}

ANSWER_KEYS = {'answer_col_ids', 'mask_token_positions', 'normalized_answers', 'answer_dep_a_start', \
               'answer_dep_qid', 'answer_end', 'answer_start', 'answer_qid'}

class EleetDataset(Dataset):
    """A simple dataset where data is stored in a h5 file and uncompressed before use."""

    def __init__(self, h5group, limit: int = None, offset: int=0,
                 fraction: float = None, offset_fraction: float = 0.):
        """Initialize the dataset."""
        super().__init__()
        self.encodings = h5group
        limit = limit or float("inf")
        l = len(self.encodings["input_ids"])
        self.limit = min(int(fraction * l), limit) if fraction is not None else limit
        self.offset = max(int(offset_fraction * l), offset)
        self.cast = torch.tensor if not isinstance(h5group["input_ids"], torch.Tensor) else (lambda x: x)

    @staticmethod
    @contextmanager
    def uncompress(dataset, *splits, return_compressed=False):
        q_keys = ('answer_col_ids', 'answer_dep_a_start', 'answer_dep_qid', 'answer_end', 'answer_qid', 'answer_start',
                  'header_query_col_ids', 'header_query_end', 'header_query_start', 'mask_token_positions',
                  'normalized_answers')
        (Path.home() / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path.home() / ".tmp") as tdir:
            with h5py.File(dataset, "r") as h5file:
                with h5py.File(Path(tdir) / "temp_h5dir.h5", "w") as f:
                    for split in splits:
                        group = f.require_group(split)
                        num_normal_queries = ((np.array(h5file[split]["answer_col_ids"]) != -1).reshape(
                            -1, h5file[split]["answer_col_ids"].shape[-1]).sum(0) > 0).sum()
                        num_header_queries = ((np.array(h5file[split]["header_query_col_ids"]) != -1).reshape(
                            -1, h5file[split]["header_query_col_ids"].shape[-1]).sum(0) > 0).sum()
                        num_queries = max(num_normal_queries, num_header_queries)
                        print("Max num queries:", num_queries)
                        for k, v in tqdm(h5file[split].items(), desc=f"Uncompressing traing data for split {split}",
                                         total=len(h5file[split])):
                            print(k, str(v), file=stdout)
                            if k in q_keys:
                                v = v[:, :, :num_queries]
                            group.create_dataset(k, data=v)
                            stdout.flush()

                with h5py.File(Path(tdir) / "temp_h5dir.h5", "r") as f:
                    if return_compressed:
                        yield h5file, f
                    else:
                        yield f


    def __getitem__(self, idx):
        """Get an element from the dataset."""
        result = dict(
            idx=idx + self.offset,
            table_id=torch.tensor([int(x) for x in self.encodings["origin"][idx + self.offset, 0].split(b"-")]),
            **{key: self.cast(val[idx + self.offset])  # pylint:  disable=E1102
            for key, val in self.encodings.items() if key not in IGNORE_KEYS}
        )
        if "text_ids" in self.encodings:
            result["text_ids"] = torch.tensor([int(x.replace(b"Q", b"").replace(b"-", b""))
                                               for x in self.encodings["text_ids"][idx + self.offset]])
        return result

    def __len__(self):
        """Compute the number of elements."""
        return min(self.limit, len(self.encodings["input_ids"]) - self.offset)

    @property
    def origin(self):
        return self.encodings["origin"]


class EleetPreTrainingDataset(EleetDataset, IterableDataset):
    """A Dataset that loads data in batches from h5 to reduce decompression overhead. Used for large datasets,
    where decompression beforehand is not feasible, e.g. for pretraining."""

    def __init__(self, *args, num_uncompress=1000, **kwargs):
        IterableDataset.__init__(self)
        EleetDataset.__init__(self, *args, **kwargs)
        self.total_num_samples = min(self.encodings["input_ids"].shape[0] - self.offset, self.limit)
        self.num_uncompress = num_uncompress
        self.chunk_size = self.encodings["input_ids"].chunks[0]
        self.num_chunks_per_group = ceil(self.num_uncompress / self.chunk_size)
        self.total_num_groups = ceil(self.total_num_samples / (self.chunk_size * self.num_chunks_per_group))

    def __iter__(self):
        worker_info = get_worker_info()
        self.num_partitions = worker_info.num_workers
        self.num_groups_in_partition = ceil(self.total_num_groups / worker_info.num_workers)
        self.group_range = range(worker_info.id * self.num_groups_in_partition,
                                 (worker_info.id + 1)* self.num_groups_in_partition)
        for group_id in self.group_range:
            sample_slice = slice(group_id * self.num_chunks_per_group * self.chunk_size + self.offset,
                                 (group_id + 1) * self.num_chunks_per_group * self.chunk_size + self.offset)
            yield from self.fetch(sample_slice)

    def fetch(self, sample_slice):
        this_encodings = {k: v[sample_slice] for k, v in self.encodings.items()}
        order = torch.randperm(this_encodings["input_ids"].shape[0])
        for i in order:
            result = dict(
                idx=i + sample_slice.start,
                table_id=torch.tensor([int(x) for x in this_encodings["origin"][i, 0].split(b"-")]),
                **{key: self.cast(val[i])  # pylint:  disable=E1102
                   for key, val in this_encodings.items() if key not in IGNORE_KEYS}
            )
            if "text_ids" in this_encodings:
                result["text_ids"] = torch.tensor([int(x.replace(b"Q", b"").replace(b"-", b""))
                                                   for x in this_encodings["text_ids"][i]])
            yield result


class EleetInferenceDataset(EleetDataset):
    """A dataset used for inference."""

    def __init__(self, *args, **kwargs):
        """Initialize the dataset."""
        super().__init__(*args, **kwargs)
        self._load()
    
    def _load(self):
        self.table_ids = torch.tensor([[int(x) for x in x.split(b"-")]
                                       for x in np.array(self.encodings["origin"][:, 0])])
        self.max_num_answers = ((self.encodings["answer_end"] > 0).sum((0, 1)) > 0).sum()
        self.encodings = {k: (self.encodings[k] if k not in ANSWER_KEYS
                          else self.encodings[k][:, :, :self.max_num_answers])
                          for k in self.encodings if k not in IGNORE_KEYS}
        _, self.selector, counts = torch.unique(self.table_ids, dim=0, return_inverse=True, return_counts=True)
        self.max_num_windows = counts.max()
        self.max_num_answers = ((self.encodings["answer_end"] > 0).sum((0, 1)) > 0).sum()

    def __getitem__(self, idx):
        """Get an element from the dataset."""
        mask = self.selector == idx + self.offset
        r = dict(
            idx=torch.where(mask)[0].view(-1, 1),
            table_id=self.table_ids[mask],
            **{key: val[mask]  # pylint:  disable=E1102
               for key, val in self.encodings.items()}
        )
        r_ex = {
            k: torch.full((self.max_num_windows, *v.size()[1:]), fill_value=-1, dtype=v.dtype, device=v.device)
                          for k, v in r.items()
        }
        for k in r.keys():
            r_ex[k][:r[k].size(0)] = r[k]
        return r_ex

    def __len__(self):
        """Compute the number of elements."""
        return int(min(self.limit, float(self.selector.max() + 1 - self.offset)))

    @property
    def origin(self):
        return self.encodings["origin"]
