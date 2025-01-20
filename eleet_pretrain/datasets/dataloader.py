import torch
import logging
from torch.utils.data import Dataset, BatchSampler, SequentialSampler


logger = logging.getLogger(__name__)




class FastModeBatchSampler(BatchSampler):
    """
    A batch sampler that allows to first perform the first iteration of the complex db ops algorithm on all
    database, and then advance all datapoints at once.
    """
    def __init__(self, col_ids, batch_size, drop_last=False, shuffle=None):
        if shuffle or drop_last:
            raise NotImplementedError
        self.col_ids = col_ids
        self.batch_samplers = []
        for v in col_ids.unique():
            sampler = SequentialSampler(col_ids[col_ids == v])
            self.batch_samplers.append((v, BatchSampler(sampler, batch_size, drop_last=False)))

    def __iter__(self):
        for col_id, bs in self.batch_samplers:
            sub = torch.where(self.col_ids == col_id)[0]
            for i in bs:
                yield sub[i].tolist()
        

    def __len__(self):
        return sum(len(bs) for _, bs in self.batch_samplers)


class TempBatchedDataset(Dataset):
    def __init__(self, current_data):
        self.current_data = current_data

    def __getitem__(self, idx):
        """Get an element from the dataset."""
        result = {
            key: val[idx] for key, val in self.current_data.items()
        }
        return result

    def __len__(self):
        """Compute the number of elements."""
        return 0 if "input_ids" not in self.current_data else len(self.current_data["input_ids"])
