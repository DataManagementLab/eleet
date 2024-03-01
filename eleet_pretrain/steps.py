"""Organize different steps for e.g. data preparation."""

import argparse
import logging
from pathlib import Path
from typing import Set, Type

logger = logging.getLogger(__name__)


class Step():
    """One step in loading the dataset."""

    depends_on: Set[Type["Step"]] = set()

    def check_done(self, args, **kwargs):
        """Check whether the step has already been executed."""

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add the arguments of the parser."""

    def run(self, args, **kwargs):
        """Execute the step."""

    @staticmethod
    def logging_setup(args, **kwargs):  # pylint: disable=unused-argument
        """Set up logging. Return a method to be called when procedure is finished."""
        return NotImplemented


def dummy_finished_handler(exception):  # pylint: disable=unused-argument
    """A dummy handler, that does nothing when program is finished."""
    return None


def run_steps(steps, **kwargs):
    """Run the steps for loading a dataset."""
    executed = set()

    def func(args):
        finished_handler = dummy_finished_handler
        for step in steps:
            x = type(step).logging_setup(args, **kwargs)
            if x != NotImplemented:
                if isinstance(x, tuple):
                    kwargs_update, finished_handler = x
                    kwargs.update(kwargs_update)
                else:
                    finished_handler = x
                break

        for step in steps:
            if not step.check_done(args, **kwargs) or step.depends_on & executed:
                executed.add(type(step))
                logger.info(f"Running step {type(step).__name__}")
                try:
                    step.run(args, **kwargs)
                except (Exception, KeyboardInterrupt) as e:
                    finished_handler(e)
                    raise e

            else:
                logger.info(f"Skipping step {type(step).__name__}")

        finished_handler(None)
    return func
