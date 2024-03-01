import argparse
from contextlib import contextmanager
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import subprocess
from attr import define, field
from eleet.methods.text_to_table.engine import T2TEngine


logger = logging.getLogger(__name__)


@define
class T2TFinetuningEngine(T2TEngine):
    args = field(init=False)

    def __attrs_post_init__(self):
        self.args = self.parse_args()

    def finetune(self, dataset_name, finetuning_inputs, valid_inputs=None, callbacks=[]):
        max_split_size = self.get_max_split_size(finetuning_inputs)
        finetune_split_sizes = sorted(set(min(x, max_split_size) for x in self.args.finetune_split_sizes), reverse=True)
        now = datetime.now()
        date_prefix = now.strftime("%Y-%m-%d_%H-%M-%S")
        this_dir = Path(f"models/{dataset_name}/text_to_table/{date_prefix}").absolute()
        this_dir.mkdir(exist_ok=True, parents=True)
        for split_size in map(int, finetune_split_sizes):
            self._finetune_with_split_size(dataset_name, finetuning_inputs, valid_inputs, callbacks,
                                           this_dir=this_dir, split_size=split_size)

    def get_max_split_size(self, finetuning_inputs):
        max_split_size = []
        for p in Path(finetuning_inputs).iterdir():
            if not p.name.startswith("train.data"):
                continue
            with p.open() as f:
                max_split_size += [len(f.readlines())]
        return max(max_split_size)

    def _finetune_with_split_size(self, dataset_name, finetuning_inputs, valid_inputs, callbacks, this_dir, split_size):
        with self.get_train_split(finetuning_inputs, valid_inputs, split_size) as train_split:
            logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.DEBUG)

            checkpoint_dir = this_dir / f"checkpoints.{train_split.name}"
            checkpoint_dir.mkdir(parents=True)

            os.chdir("text_to_table")
            
            logger.info(f"Running training on split of size {train_split.name}.")
            subprocess.run(["bash", "scripts/preprocess.sh", str(train_split), str(self.model_path)])        
            subprocess.run(["bash", "../scripts/train_t2t_had.sh", str(train_split), str(self.model_path),
                            str(checkpoint_dir)])

            for f in checkpoint_dir.iterdir():
                if "checkpoint_average_best" not in f.name and f.name != "log":
                    logger.info(f"Removing {f}")
                    os.remove(f)

            os.chdir("..")

    @contextmanager
    def get_train_split(self, finetuning_inputs, valid_inputs, split_size):
        split_dir = Path(finetuning_inputs) / str(split_size)
        split_dir.mkdir()
        for file in Path(finetuning_inputs).iterdir():
            if file.is_dir():
                continue
            with file.open() as f_out:
                with (split_dir / file.name.split("---")[0]).open("a") as f_in:
                    for i, line in enumerate(f_out):
                        if i >= split_size:
                            break
                        print(line, file=f_in, end="")

        for file in Path(valid_inputs).iterdir():
            with file.open() as f_out:
                with (split_dir / file.name.split("---")[0]).open("a") as f_in1:
                    with (split_dir / file.name.replace("valid", "test").split("---")[0]).open("a") as f_in2:
                        for line in f_out:
                            print(line, file=f_in1, end="")
                            print(line, file=f_in2, end="")
        yield split_dir.absolute()

    def parse_args(self):
        """Add the arguments of the parser."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--finetune-split-sizes", type=int, nargs="+", default=[(2 ** i) for i in range(7, 15)])
        args = parser.parse_args()
        return args
