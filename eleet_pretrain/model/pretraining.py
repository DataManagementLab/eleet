"""Train the model."""

import argparse
import logging
from pathlib import Path

import gdown
import torch
import h5py
import multiprocessing
from transformers import TrainingArguments
from eleet_pretrain.datasets.dataset import EleetPreTrainingDataset, EleetDataset
from eleet_pretrain.model.collator import EleetDataCollator, PRETRAINING_OBJECTIVE
from eleet_pretrain.steps import Step
from eleet_pretrain.model.base_model import BaseEleetModel
from eleet_pretrain.utils import insert_into_global_log_begin, insert_into_global_log_end, logging_setup
from eleet_pretrain.metrics import compute_metrics
from eleet_pretrain.model.training_utils import get_logging_path, get_target_path, get_checkpoint_path, \
    get_preprocessed_dataset, logging_begin

logger = logging.getLogger(__name__)


class TrainingStep(Step):
    """Download pre-trained weights."""

    def run(self, args, logging_path, target_path):
        """Execute the step."""
        # Create directories
        checkpoint_path = get_checkpoint_path(args)
        dataset = get_preprocessed_dataset(args.dataset)
        disabled_pretraining_objectives = self.get_disabled_pretraining_objectives(args)

        with EleetDataset.uncompress(dataset, args.eval_split,
                                    return_compressed=True) as (h5file_comp, h5file):
            train_dataset = EleetPreTrainingDataset(
                h5file_comp[args.train_split], limit=args.train_split_limit, offset=args.train_split_offset,
                fraction=args.train_split_fraction, offset_fraction=args.train_split_offset_fraction
            )
            valid_dataset = EleetDataset(
                h5file[args.eval_split], limit=args.eval_split_limit, offset=args.eval_split_offset,
                fraction=args.eval_split_fraction, offset_fraction=args.eval_split_offset_fraction
            )

            model = BaseEleetModel.from_pretrained(**vars(args))
            model.set_debug_fraction(args.debug_fraction)

            num_devices = max(1, torch.cuda.device_count())
            gradient_accumulation_steps = int(args.train_batch_size / (args.per_device_train_batch_size * num_devices))

            logging_begin(model, train_dataset, valid_dataset)
            training_args = TrainingArguments(
                num_train_epochs=args.num_train_epochs,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                dataloader_num_workers=args.dataloader_num_workers,

                output_dir=checkpoint_path,
                logging_dir=logging_path,

                logging_strategy="steps",
                logging_steps=args.logging_steps,
                save_strategy="steps",
                # save_total_limit=2,
                save_steps=args.save_steps,

                evaluation_strategy="steps",
                eval_steps=args.eval_steps,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                eval_accumulation_steps=args.eval_accumulation_steps,
                label_names=["query_labels", "query_coords", "masked_context_token_labels"],
                prediction_loss_only=False,
                remove_unused_columns=False,

                ddp_find_unused_parameters=True,
                fp16=torch.cuda.is_available(),
            )

            trainer = model.get_trainer(
                model=model,
                data_collator=EleetDataCollator(model.config, model.tokenizer, disabled=disabled_pretraining_objectives,
                                               model=model, is_pretraining=True),
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=lambda x: compute_metrics(x, logging_path, args.eval_split)
            )
            resume = args.resume and any("checkpoint" in x.name for x in checkpoint_path.iterdir())
            # torch.autograd.set_detect_anomaly(True)
            trainer.train(True if resume else None)
            model.save_pretrained(target_path)

    @staticmethod
    def logging_setup(args):
        logging_path = get_logging_path(args)
        target_path = get_target_path(args)
        checkpoint_path = get_checkpoint_path(args)
        logging_setup(args.log_level, log_file=logging_path / "pretraining.log")
        training_log_file = Path(args.model_dir / "training.log")
        start_time = insert_into_global_log_begin(training_log_file,
                                                  logging_path=logging_path,
                                                  checkpoint_path=checkpoint_path,
                                                  target_path=target_path / "pytorch_model.bin")
        return {"logging_path": logging_path, "target_path": target_path}, \
            lambda exception: insert_into_global_log_end(training_log_file, start_time, exception)

    def get_disabled_pretraining_objectives(self, args):
        disabled = args.disable_pretraining_objectives
        if disabled is None:
            return [PRETRAINING_OBJECTIVE.RELEVANT_TEXT_DETECTION]
        disabled = [getattr(PRETRAINING_OBJECTIVE, x.upper().replace("-", "_")) for x in disabled]
        if PRETRAINING_OBJECTIVE.RELEVANT_TEXT_DETECTION not in disabled:
            disabled.append(PRETRAINING_OBJECTIVE.RELEVANT_TEXT_DETECTION)
        return disabled

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add the arguments of the parser."""
        parser.add_argument("--dataset", type=Path, default=Path(__file__).parents[2] / "datasets",
                            help="Path to dataset HDF file or to root directory of datasets. "
                            "Latter will choose the latest dataset in the directory.")
        parser.add_argument("--model-name-or-path", type=str, default="tabert_base_k3")
        parser.add_argument("--model-dir", type=Path, default=Path(__file__).parents[2] / "models",
                            help="Root directory of models.")
        parser.add_argument("--resume", type=Path, help="The checkpoint directory of the run to resume.")
        parser.add_argument("--local-rank", type=int, default=0)
        parser.add_argument("--dataloader-num-workers", type=int, default=int(multiprocessing.cpu_count()) - 1)
        parser.add_argument("--debug-fraction", type=float, default=1/1000)
        parser.add_argument("--disable-vertical-transform", action="store_true")

        parser.add_argument("--num-train-epochs", type=int, default=3)
        parser.add_argument("--learning-rate", type=float, default=5e-05)
        parser.add_argument("--warmup-ratio", type=float,  default=0.1)
        parser.add_argument("--per-device-train-batch-size", type=int, default=6)
        parser.add_argument("--train-batch-size", type=int, default=256)
        parser.add_argument("--weight-decay", type=float, default=0.01)
        parser.add_argument("--max-grad-norm", type=float, default=1.0)
        parser.add_argument("--eval-steps", type=int, default=50_000),
        parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
        parser.add_argument("--eval-accumulation-steps", type=int, default=16)
        parser.add_argument("--logging-steps", type=int, default=10_000)
        parser.add_argument("--save-steps", type=int, default=10_000)

        parser.add_argument("--train-split", type=str, default="train_default")
        parser.add_argument("--train-split-limit", type=int, default=None)
        parser.add_argument("--train-split-offset", type=int, default=0)
        parser.add_argument("--train-split-fraction", type=float, default=None)
        parser.add_argument("--train-split-offset-fraction", type=float, default=0.)
        parser.add_argument("--eval-split", type=str, default="development_default")
        parser.add_argument("--eval-split-limit", type=int, default=30_000)
        parser.add_argument("--eval-split-offset", type=int, default=0)
        parser.add_argument("--eval-split-fraction", type=float, default=None)
        parser.add_argument("--eval-split-offset-fraction", type=float, default=0.)

        parser.add_argument("--sd-loss-multiplier", type=float, default=300.)
        parser.add_argument("--mlm_loss_multiplier", type=float, default=1.)
        parser.add_argument("--hq-loss-multiplier", type=float, default=80.)
        parser.add_argument("--rt-loss-multiplier", type=float, default=80.)
        parser.add_argument("--dup-loss-multiplier", type=float, default=1.)
        parser.add_argument("--cls-loss-multiplier", type=float, default=1.)

        parser.add_argument("--disable-pretraining-objectives", type=str, nargs="+", choices=[
            x.lower().replace("_", "-") for x in vars(PRETRAINING_OBJECTIVE).keys() if not x.startswith("__")
            if x != "RELEVANT_TEXT_DETECTION"
        ])

class DownloadPreTrainedStep(Step):
    """Download pre-trained weights."""

    def check_done(self, args, **kwargs):
        """Check whether the step has already been executed."""
        return (((args.model_dir) / "base-models" / "tabert_base_k3").exists()
                and ((args.model_dir) / "base-models" / "tabert_large_k3").exists())

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add the arguments of the parser."""
        parser.add_argument("--model-dir", type=Path, default=Path(__file__).parents[2] / "models",
                            help="Root directory of datasets.")

    def run(self, args, **kwargs):
        """Execute the step."""
        pretrained_dir = (args.model_dir) / "base-models"
        pretrained_dir.mkdir(exist_ok=True, parents=True)
        if not ((args.model_dir) / "base-models" / "tabert_base_k3").exists():
            gdown.cached_download(url="https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj",
                                  path=str(pretrained_dir / "tabert_base_k3.tar.gz"), postprocess=gdown.extractall)
        if not ((args.model_dir) / "base-models" / "tabert_large_k3").exists():                   
            gdown.cached_download(url="https://drive.google.com/uc?id=17NTNIqxqYexAzaH_TgEfK42-KmjIRC-g",
                                  path=str(pretrained_dir / "tabert_large_k3.tar.gz"), postprocess=gdown.extractall)
