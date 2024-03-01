from collections import namedtuple
from contextlib import ExitStack, contextmanager
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
from torch.multiprocessing import Queue
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional

from attr import field
import numpy as np
from torch.multiprocessing import Process
import tqdm
from eleet.database import Database
from eleet.methods.base_preprocessor import BasePreprocessor
from attrs import define


logger = logging.getLogger(__name__)
DELIMITER = ";"
MULTI_VALUE_SEPARATOR = ", "
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


ModelInput = namedtuple("ModelInput", ["prompts", "data", "table_name", "evidence_columns",
                                       "num_samples", "finetune_split_size"])
REQUEST_FIRST = True


@define
class PromptLoader():
    prompts: Callable = field()
    operations: List[Callable] = field(init=False, default=[])
    num_processes: int = field(default=1)
    num_prompts: int = field(default=2**32)

    def __iter__(self):
        with TemporaryDirectory(dir=".", prefix="promptloader") as tempdir:
            num_processes = self.num_processes
            queues = [Queue(maxsize=20) for _ in range(num_processes)]
            processes = []
            for i in range(num_processes):
                process = Process(
                        target=self._iter_prompts,
                        kwargs={"num_processes": num_processes,
                                "process_id": i, "queue": queues[i], "tempdir": tempdir}, daemon=True)
                process.start()
                processes.append(process)

            for i in tqdm.tqdm(range(self.num_prompts), desc="Generate Prompts"):
                queue = queues[i % num_processes]
                item = queue.get()
                if item is None:
                    break
                with open(item, "rb") as f:
                    item = pickle.load(f)
                yield item

            for process in processes:
                process.join()

    def iter_distributed(self, num_processes, process_id):
        for prompt in self.prompts(num_processes=num_processes, process_id=process_id):
            for operation in self.operations:
                p, prefix = prompt
                p = operation(p)
                prompt = p, prefix
            yield prompt

    def _iter_prompts(self, num_processes, process_id, queue, tempdir):
        for i, prompt in enumerate(tqdm.tqdm(self.iter_distributed(num_processes, process_id),
                                             desc=f"Gather Prompts {process_id}", position=process_id + 1)):
            file = Path(tempdir) / f"prompt_{process_id}_{i}.pkl"
            with open(file, "wb") as f:
                pickle.dump(prompt, f)
                queue.put(str(file.absolute()))
        queue.put(None)


@define
class LLMPreprocessor(BasePreprocessor):
    train_db: Optional[Database] = field()
    num_samples: int = field()
    finetune_split_size: int = field()
    rng: np.random.BitGenerator = field(init=False, default=np.random.default_rng(42))

    @contextmanager
    def compute_model_input(self, data, report_column, report_table_name, extract_attributes, identifying_attribute,
                            example_rows, multi_table, limit=2**32, index_mode: bool=False):
        evidence_columns = self.get_evidence_columns(data, report_column)
        if identifying_attribute is not None and identifying_attribute not in extract_attributes:
            extract_attributes = [identifying_attribute] + extract_attributes
        textual_operand = data.data[report_column]
        num_prompts = len([x for x in textual_operand.index.unique(level=0) if x < limit])
        prompts = partial(self.generate_prompts, table_name=report_table_name,
                          report_column=report_column, textual_operand=textual_operand,
                          columns=extract_attributes, limit=limit)
        yield ModelInput(prompts=PromptLoader(prompts, num_processes=15, num_prompts=num_prompts),
                         data=data.data.iloc[data.data.index.get_level_values(0) < limit],
                         table_name=report_table_name,
                         evidence_columns=evidence_columns,
                         num_samples=self.num_samples,
                         finetune_split_size=self.finetune_split_size)


    @contextmanager
    def compute_finetuning_data(self, data, split, limit=2**32):
        data = sorted(data, key=lambda x: len(x.labels.normed))
        grouped_data = {d.full_name[0]: [dd for dd in data if dd.full_name[0] == d.full_name[0]] for d in data}
        with TemporaryDirectory(dir=".", prefix="gpt_tmp") as tempdir:
            with ExitStack() as stack:
                for group in grouped_data.values():
                    text_collection = group[0].text_collection
                    pbar = tqdm.tqdm(desc="Generate Training Data",
                                     total=len(text_collection.text_tables) * len(text_collection.data))
                    for text_table_name, text_table in text_collection.text_tables.items():
                        f = stack.enter_context(
                            open(Path(tempdir) / f"{split}---{text_collection.name}---{text_table_name}", "w"))
                        columns = text_table.labels.normed.columns.tolist()
                        prompts = self.generate_prompts(text_table.full_name,
                                                        text_collection.text_column,
                                                        text_collection.data[text_collection.text_column],
                                                        columns, limit=limit)

                        labels = self.apply_multi_value_separator(text_table.labels.normed)[columns]
                        for i, p in enumerate(prompts):
                            label = labels.loc[[i]] if i in labels.index else labels.loc[[]]
                            label = [{"role": "assistant",
                                    "content": f"Output:\n{label.to_csv(sep=DELIMITER, index=False)}"}]
                            prompt = p[0] + label
                            as_json = json.dumps({"messages": prompt})
                            print(as_json, file=f)
                            pbar.update(1)
            yield tempdir

    def generate_prompts(self, table_name, report_column, textual_operand, columns, limit=2**32,
                         num_processes=1, process_id=0):
        few_shot_labels = None
        if self.num_samples > 0:
            assert self.train_db is not None, "Cannot generate few-shot prompt without train_db"
            few_shot_labels = self.train_db.texts[table_name[0]].text_tables[table_name[1]].labels
            if few_shot_labels is None:
                raise ValueError(f"No labels found for {table_name[0]} {table_name[1]}. "
                                 "Labels are required for few-shot prompt.")
            few_shot_labels = self.apply_multi_value_separator(few_shot_labels.normed)
            if few_shot_labels.index.names is None:
                few_shot_labels = few_shot_labels.reset_index().set_index("index")
            few_shot_labels = few_shot_labels[columns]

        for i in sorted(textual_operand.index.unique(level=0))[process_id::num_processes]:
            if i >= limit:
                break
            text = textual_operand.loc[[i]].iloc[0]
            prompt, prefix = self.single_prompt(columns, table_name, text)
            if self.num_samples > 0:
                prompt = self.generate_few_shot_prompt(report_column, columns, table_name, prompt, few_shot_labels)
            yield prompt, prefix

    def generate_few_shot_prompt(self, report_column, columns, table_name, prompt, few_shot_labels):
        if self.train_db is None:
            raise ValueError("Cannot generate few-shot prompt without train_db")
        texts = self.train_db.texts[table_name[0]].data[report_column]

        # Random selection of few-shot samples
        rng = np.random.default_rng(int(hashlib.md5(prompt[0]["content"].encode("utf-8")).hexdigest(), 16) % 1000)
        finetune_split_size = self.finetune_split_size if self.finetune_split_size != "full" else (2 ** 32)
        finetune_split_size = min(finetune_split_size, len(few_shot_labels.index.unique(level=0)))
        num_samples = min(self.num_samples, finetune_split_size)
        prompt_report_ids_random = rng.choice(np.array(few_shot_labels.index.unique(level=0)[:finetune_split_size]),
                                              num_samples, replace=False)
        # Generate prompts
        result = []
        for i in prompt_report_ids_random[-num_samples:]:
            text = texts.loc[i]
            label = few_shot_labels.loc[[i]] if i in few_shot_labels.index else few_shot_labels.loc[[]]
            label = label.to_csv(sep=DELIMITER, index=False)
            result += self.single_prompt(columns, table_name, text)[0]
            result += [{"role": "assistant", "content": f"Output:\n{label}"}]
        return result + prompt

    def apply_multi_value_separator(self, labels):
        return labels.applymap(lambda x: f"{MULTI_VALUE_SEPARATOR} ".join(sorted(set(x))))

    def single_prompt(self, columns, table_name, text):
        prefix = f"{DELIMITER.join(columns)}\n"
        result = [{"role": "user", "content":
            (f"Request: Transform the input text to a {table_name[1]} table. "
             f"Only output the table in CSV-Format without explanations. Use '{DELIMITER}' as CSV delimiter. "
             f"If there are multiple values for a cell, use '{MULTI_VALUE_SEPARATOR}' as value delimiter. "
             f"If there is no information for a cell, leave it empty. The header row is: {prefix}\n"
             f"Input: {text}"
            ) if REQUEST_FIRST else
            (f"Input: {text}\n"
             f"Request: Transform the input text to a {table_name[1]} table. "
             f"Only output the table in CSV-Format without explanations. Use '{DELIMITER}' as CSV delimiter. "
             f"If there are multiple values for a cell, use '{MULTI_VALUE_SEPARATOR}' as value delimiter. "
             f"If there is no information for a cell, leave it empty. The header row is: {prefix}")
        }]
        return result, prefix
