"""Training script."""

import argparse
import logging
from eleet_pretrain.steps import run_steps
from eleet_pretrain.model.pretraining import DownloadPreTrainedStep, TrainingStep


if __name__ == '__main__':
    steps = [DownloadPreTrainedStep(), TrainingStep()]
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--log-level", type=lambda x: getattr(logging, x.upper()), default=logging.INFO)
    for step in steps:
        step.add_arguments(parser)
    args = parser.parse_args()
    run_steps(steps)(args)
