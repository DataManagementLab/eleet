import argparse
import logging
from math import ceil
import os
from pathlib import Path
from attr import define, field
import torch
from transformers import Trainer, TrainingArguments
from eleet_pretrain.model.training_utils import get_checkpoint_path, get_logging_path, get_model_name_or_path, get_target_path_finetuning, logging_begin
from eleet_pretrain.utils import insert_into_global_log_begin, insert_into_global_log_end, logging_setup
from eleet.methods.base_engine import BaseEngine
from eleet.methods.eleet.data_collator import ELEETFinetuningCollator
from eleet.methods.eleet.dataset import ELEETFinetuningDataset
from eleet.methods.eleet.engine import ELEETEngine
from eleet.methods.eleet.model import ELEETFinetuningModel
from torch.multiprocessing import spawn


logger = logging.getLogger(__name__)


@define
class ELEETFinetuneEngine(ELEETEngine):
    num_preprocessing_workers_per_gpu: int = field(init=False, default=None)  # not used
    num_postprocessing_workers_per_gpu: int = field(init=False, default=None)  # not used
    raise_exceptions: bool = field(default=False)
    logging_path: Path = field(init=False)
    target_path: Path = field(init=False)
    args = field(init=False)
    model_name_or_path: str = field(init=False)
    output_dir: Path = field(init=False)

    def __attrs_post_init__(self):
        self.args = self.parse_args()
        self.model_name_or_path = self.args.model_name_or_path
        if self.model_name_or_path == "latest":
            self.model_name_or_path = get_model_name_or_path(self.args.model_dir)
        self.model = ELEETFinetuningModel.from_pretrained(**vars(self.args))
        self.model.set_debug_fraction(self.args.debug_fraction)
        self.model.freeze_layers(self.args.freeze_num_layers)
        self.model.share_memory()

    def finetune(self, dataset_name, finetuning_inputs, valid_inputs=None, callbacks=[]):
        self.output_dir = get_checkpoint_path(self.args, dataset=dataset_name, method="ours")
        self.logging_path = get_logging_path(self.args, dataset=dataset_name, method="ours")
        self.target_path = get_target_path_finetuning(self.args, dataset=dataset_name, method="ours")

        logging_setup(self.args.log_level, log_file=self.logging_path / "logging.log")
        exception = None
        finetuning_log_file = Path(self.args.model_dir / "finetuning.log")
        start_time = insert_into_global_log_begin(finetuning_log_file, self.logging_path, self.output_dir)
        try:
            self._finetune(finetuning_inputs=finetuning_inputs, valid_inputs=valid_inputs, callbacks=callbacks)
        except BaseException as e:
            exception = e
            if self.raise_exceptions:
                raise e
        finally:
            insert_into_global_log_end(finetuning_log_file, start_time, exception)

    def _finetune(self, finetuning_inputs, valid_inputs=None, callbacks=[], use_port=None):
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        if use_port is not None or "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(use_port) if use_port is not None else "29500"
        os.environ["USE_FP16"] = str(int(torch.cuda.is_available()))

        max_split_size = max(x.max_split_size for x in finetuning_inputs)
        finetune_split_sizes = sorted(set(min(x, max_split_size) for x in self.args.finetune_split_sizes), reverse=True)
        for split_size in map(int, finetune_split_sizes):
            spawn(self._finetune_with_split_size, args=(finetuning_inputs, valid_inputs, callbacks, split_size),
                  nprocs=torch.cuda.device_count())

    def _finetune_with_split_size(self, index, finetuning_inputs, valid_inputs, callbacks, split_size):
        os.environ["LOCAL_RANK"] = str(index)
        os.environ["RANK"] = str(index)

        train_dataset = ELEETFinetuningDataset(finetuning_inputs, split_size=split_size, shuffle=True)
        valid_dataset = ELEETFinetuningDataset(valid_inputs, split_size=self.args.eval_split_limit, shuffle=False) \
                if valid_inputs is not None else None

        num_devices = max(torch.cuda.device_count(), 1)
        gradient_accumulation_steps = self.get_accumulation_steps(self.args, num_devices)
        num_steps, eval_steps = self.get_num_steps(self.args, split_size, num_devices, gradient_accumulation_steps)

        data_collator = ELEETFinetuningCollator(self.model.config, self.model.tokenizer)

        logging_begin(self.model, train_dataset, valid_dataset)
        training_args = TrainingArguments(
                max_steps=num_steps,
                learning_rate=self.args.learning_rate,
                lr_scheduler_type=self.args.learning_rate_schedule.lower(),
                warmup_ratio=self.args.warmup_ratio,
                per_device_train_batch_size=self.args.per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=self.args.weight_decay,
                max_grad_norm=self.args.max_grad_norm,
                dataloader_num_workers=self.args.dataloader_num_workers,
                output_dir=str(self.output_dir / str(split_size) if self.output_dir else Path(".")),
                logging_dir=str(self.logging_path / str(split_size)),
                logging_strategy="steps",
                logging_steps=self.args.logging_steps,
                save_strategy="steps",
                save_total_limit=2,
                save_steps=self.args.save_steps,
                evaluation_strategy="steps",
                eval_steps=eval_steps if valid_dataset is not None else 2 ** 64,
                per_device_eval_batch_size=self.args.per_device_eval_batch_size,
                eval_accumulation_steps=self.args.eval_accumulation_steps,
                label_names=["query_labels", "header_query_labels", "deduplication_labels"],
                prediction_loss_only=False,
                metric_for_best_model="sd_f1",
                remove_unused_columns=False,
                ddp_find_unused_parameters=True,
                local_rank=index,
                fp16=torch.cuda.is_available(),
            )

        trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=callbacks,
                compute_metrics=self.compute_metrics
            )
            # torch.autograd.set_detect_anomaly(True)
        if split_size > 0:
            trainer.train()
        if self.args.num_eval > 0 and valid_dataset is not None:
            trainer.evaluate()
        if self.target_path is not None:
            self.model.save_pretrained(self.target_path / str(split_size))

    def compute_metrics(self, preds_labels):
        return dict(
            sd_loss=preds_labels.predictions[0].mean(),
            hq_loss=preds_labels.predictions[1].mean(),
            dup_loss=preds_labels.predictions[2].mean(),
            sd_f1=preds_labels.predictions[3].mean(),
            hq_f1=preds_labels.predictions[4].mean(),
            dup_f1=preds_labels.predictions[5].mean()
        )

    def get_accumulation_steps(self, args, num_devices):
        gradient_accumulation_steps = int(ceil(args.train_batch_size / (
                    args.per_device_train_batch_size * num_devices)))
        return gradient_accumulation_steps

    def get_num_steps(self, args, split_size, num_devices, gradient_accumulation_steps):
        batch_size = max(1, gradient_accumulation_steps * args.per_device_train_batch_size * num_devices)
        steps_per_epoch = int(ceil(split_size / batch_size))
        num_steps = min(steps_per_epoch * args.num_train_epochs, args.max_steps)
        if args.num_eval > 0:
            eval_steps = max(1, int(ceil(num_steps / args.num_eval)))
        else:
            eval_steps = 2 ** 32
        msg = f"Will run evaluation every {eval_steps} step(s)."
        logger.info(msg)
        print(msg)
        return num_steps, eval_steps

    def parse_args(self):
        """Add the arguments of the parser."""
        parser = argparse.ArgumentParser()
        parser.add_argument("model_name_or_path", type=str)
        parser.add_argument("--model-dir", type=Path, default=Path(__file__).parents[3] / "models",
                            help="Root directory of models.")
        parser.add_argument("--local-rank", type=int, default=0)
        parser.add_argument("--log-level", type=lambda x: getattr(logging, x.upper()), default=logging.INFO)

        parser.add_argument("--num-train-epochs", type=int, default=500)
        parser.add_argument("--max-steps", type=int, default=30_000)
        parser.add_argument("--learning-rate", type=float, default=3e-05)
        parser.add_argument("--learning-rate-schedule", type=str, choices=["cosine", "linear", "constant"], default="linear")
        parser.add_argument("--warmup-ratio", type=float,  default=0.1)
        parser.add_argument("--per-device-train-batch-size", type=int, default=6)
        parser.add_argument("--train-batch-size", type=int, default=30)
        parser.add_argument("--weight-decay", type=float, default=0.01)
        parser.add_argument("--max-grad-norm", type=float, default=1.0)
        parser.add_argument("--num-eval", type=int, default=0)
        parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
        parser.add_argument("--eval-accumulation-steps", type=int, default=16)
        parser.add_argument("--logging-steps", type=int, default=100)
        parser.add_argument("--dataloader-num-workers", type=int, default=10)
        parser.add_argument("--save-steps", type=int, default=100_000_000)
        parser.add_argument("--debug-fraction", type=float, default=1/1000)
        parser.add_argument("--freeze-num-layers", type=int, default=11)
        parser.add_argument("--eval-split-limit", type=int, default=None)
        parser.add_argument("--finetune-split-sizes", type=int, nargs="+", default=[(2 ** i) for i in range(7, 15)])

        parser.add_argument("--disable-vertical-transform", action="store_true")
        parser.add_argument("--disable-learned-deduplication", action="store_true")
        parser.add_argument("--disable-header-query-ffn-for-multi-union", action="store_true")
        parser.add_argument("--skip-store", action="store_false", dest="store")

        parser.add_argument("--sd-loss-multiplier", type=float, default=300.)
        parser.add_argument("--hq-loss-multiplier", type=float, default=80.)
        parser.add_argument("--rt-loss-multiplier", type=float, default=80.)
        parser.add_argument("--dup-loss-multiplier", type=float, default=1.)
        parser.add_argument("--cls-loss-multiplier", type=float, default=1.)

        args = parser.parse_args()
        return args
