import numpy as np
from torch.utils.data import IterableDataset



class ELEETInferenceDataset(IterableDataset):
    """A dataset used for inference."""

    def __init__(self, model, model_input_job_queue, distribute_id_col_values_queue):
        """Initialize the dataset."""
        super().__init__()
        self.model = model
        self.model_input_job_queue = model_input_job_queue
        self.distribute_id_col_values_queue = distribute_id_col_values_queue

    def __iter__(self):
        model_input = self.model_input_job_queue.get()
        model_input.set_model(self.model)
        model_input.set_distribute_id_col_values_queue(self.distribute_id_col_values_queue)

        for tensor_dict in model_input:
            yield tensor_dict


class ELEETFinetuningDataset(IterableDataset):
    """A dataset used for inference."""

    def __init__(self, finetuning_input, split_size=None, shuffle=True):
        """Initialize the dataset."""
        super().__init__()
        self.finetuning_input = finetuning_input
        self._iterators = None
        self.rng = np.random.default_rng(42)
        self.split_size = split_size
        self.shuffle = shuffle

    @property
    def iterators(self):
        if self._iterators is None:
            self._iterators = [x.iter_with_specified_split_size(self.split_size) for x in self.finetuning_input]
        return self._iterators

    def __iter__(self):
        if self.shuffle:
            yield from self._iter_shuffle()
        else:
            yield from self._iter_all()

    def _iter_shuffle(self):
        while True:
            choice = self.rng.choice(len(self.finetuning_input))
            try:
                yield next(self.iterators[choice])
            except StopIteration:
                break

    def _iter_all(self):
        for iterator in self.iterators:
            yield from iterator

    def __getstate__(self):
        return dict(
            finetuning_input=self.finetuning_input,
            split_size=self.split_size,
            shuffle=self.shuffle
        )

    def __setstate__(self, state):
        self.finetuning_input = state["finetuning_input"]
        self.split_size = state["split_size"]
        self.shuffle= state["shuffle"]
        self.rng = np.random.default_rng(42)
        self._iterators = None
