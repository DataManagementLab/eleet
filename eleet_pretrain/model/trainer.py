"""A trainer that logs components of the loss."""

from copy import deepcopy
import math
import time
from sklearn.cluster import AgglomerativeClustering
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import Trainer
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_utils import speed_metrics
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer_utils import seed_worker
from transformers.trainer_callback import ProgressCallback

from eleet_pretrain.datasets.dataloader import EleetInferenceDataLoader
from eleet_pretrain.model.model import EleetModel


class EleetTrainer(Trainer):
    """A trainer that logs components of the loss."""

    def __init__(self, *args, is_complex_operation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._additional_logs = {"num": 0}
        self.do_normalize = False
        self.is_complex_operation = is_complex_operation
        self.data_collator.is_complex_operation = is_complex_operation

    def log(self, logs):
        loss_prefix = ""
        main_loss_key = "loss"
        if main_loss_key not in logs:
            main_loss_key = "eval_loss"
            loss_prefix = "eval_"

        if main_loss_key in logs and self._additional_logs["num"] > 0:
            loss_keys = [k for k in self._additional_logs.keys() if k.endswith("_loss")]
            x = {k: (v / self._additional_logs["num"]) for k, v in self._additional_logs.items()}

            if logs[main_loss_key] == 0 or round(x["tmp-loss"] / logs[main_loss_key]) == 0:
                scaler = float("inf")
            else:
                scaler = round(x["tmp-loss"] / logs[main_loss_key])
            for k in loss_keys:
                x[k] /= scaler
            # x["weighting_penalty"] /= scaler
            del x["tmp-loss"]
            del x["num"]
            self._additional_logs = {"num": 0}
            logs.update({f"{loss_prefix}{k}": round(v, 4) for k, v in x.items()})
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_keys = [k for k in outputs.keys() if k.endswith("_loss")]
        # weight_keys = [k for k in outputs.keys() if k.endswith("_loss_weight")]
        if any(outputs[k] is not None for k in loss_keys):
            for key in loss_keys:  # "weighting_penalty", *weight_keys):
                if key not in outputs:
                    continue
                self._additional_logs[key] = self._additional_logs.get(key, 0) + (outputs[key].float().mean().item())
            self._additional_logs["tmp-loss"] = self._additional_logs.get("tmp-loss", 0) + (outputs["loss"].float().mean().item())
            self._additional_logs["num"] += 1
        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if not isinstance(self.model, EleetModel):
            return super().get_eval_dataloader(eval_dataset)

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return EleetInferenceDataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                model=self.model,
                is_complex_operation=self.is_complex_operation
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def evaluate(self, eval_dataset: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        if isinstance(self.model, EleetModel) or self.do_normalize:
            self._td_compute_metrics = self.compute_metrics
            self.compute_metrics = None

        self.model.eval_stage = 0
        output = eval_loop(  # First iteration
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if isinstance(self.model, EleetModel):
            other_labels = output.label_ids
            if len(self.label_names) == 1:
                 other_labels = (output.label_ids, )

            i = 2
            while not eval_dataloader.is_done():  # other iterations of complex db ops algorithm
                eval_dataloader.advance_stage(*map(torch.tensor, output.predictions))
                self.update_progress_bar(eval_dataloader)
                self.model.eval_stage = i
                output2 = eval_loop(
                    eval_dataloader,
                    description=f"Evaluation Table Decoder Iteration {i}",
                    # No point gathering the predictions if there are no metrics, otherwise we defer to
                    # self.args.prediction_loss_only
                    prediction_loss_only=False,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )
                output = self.merge_outputs(output, output2)
                i += 1
            eval_metrics = eval_dataloader.final_results_eval(self._td_compute_metrics, other_labels,
                                                              normalize=self.do_normalize)
            output.metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            self.compute_metrics = self._td_compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def update_progress_bar(self, eval_dataloader):
        try:
            for c in self.callback_handler.callbacks:
                if isinstance(c, ProgressCallback):
                    c.prediction_bar.total += len(eval_dataloader)
        except Exception:
            return

    def merge_outputs(self, o1, o2):
        result = EvalLoopOutput(
            predictions=o2.predictions,
            label_ids=None,
            metrics={},
            num_samples=o1.num_samples + o2.num_samples
        )
        return result

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_eval_batch_size
