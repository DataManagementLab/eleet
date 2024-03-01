from collections import namedtuple
import csv
from functools import partial
from io import StringIO
import logging
import os
from pathlib import Path
from attr import field
from llama import Llama

import numpy as np
import pandas as pd
from attrs import define
import torch
from torch.nn.functional import pad
from eleet.methods.base_engine import BaseEngine, EngineMode
from eleet.methods.llama.preprocessor import DELIMITER, MULTI_VALUE_SEPARATOR, PromptLoader
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator


logger = logging.getLogger(__name__)


@define
class LLMEngine(BaseEngine):
    max_result_tokens = 1024
    name: str = field(init=False, default="LLaMA")

    def max_prompt_length(self, x):
        raise NotImplementedError

    def finalize(self, model_input, attributes, identifying_attribute, results):
        goal_attributes = {c.lower(): c for c in attributes}
        rename = [{c: goal_attributes[c.lower()] for c in r.columns if c.lower() in goal_attributes} for r in results]
        results = [r.rename(rn, axis=1)[list(rn.values())] for r, rn in zip(results, rename)]
        if len(results) == 0:
            return self.get_empty_result(model_input.data, attributes)

        result_no_duplicate_columns = (r.loc[:, ~r.columns.duplicated()] for r in results)
        result = pd.concat(result_no_duplicate_columns, axis=0)
        result = result.fillna("").applymap(lambda x: "" if len(x) == 0 else x).sort_index()

        result.index.name = "__idx__"  # type:ignore
        if identifying_attribute is not None:
            result[identifying_attribute] = result[identifying_attribute].apply(lambda x: x[0] if len(x) > 0 else None)  # type: ignore
            result = result.loc[result[identifying_attribute].notnull() & result[identifying_attribute] != ""]  # type: ignore
            result = result.groupby([result.index.name, identifying_attribute]) \
                .agg(lambda x: x.iloc[0]).reset_index(identifying_attribute)
        else:
            result = result.groupby(result.index.names).agg(lambda x: x.iloc[0])  # type: ignore

        if identifying_attribute is not None and identifying_attribute in model_input.evidence_columns:
            index_map = {v: k for k, v in enumerate(model_input.data.index.unique(0))}
            result = model_input.data.reset_index().replace({model_input.data.index.names[0]: index_map}).rename(
                {model_input.data.index.names[0]: "__idx__"}, axis=1).merge(result, on=(
                "__idx__", identifying_attribute), how="left").drop("__idx__", axis=1)[attributes].fillna("")
        return result

    def get_num_tokens(self, text):
        raise Exception

    def get_model_name(self):
        raise Exception

    def truncate_prompt(self, prompt, get_num_tokens_func=None):
        get_num_tokens_func = get_num_tokens_func or self.get_num_tokens
        prompt_lens = [get_num_tokens_func(m["content"]) for m in prompt]
        max_len = self.max_prompt_length(self.get_model_name())
        current_len = 0
        current_prompt = []

        for p, l in zip(prompt[::-1], prompt_lens[::-1]):
            if current_len + l > max_len:
                break
            current_len += l
            current_prompt.append(p)

        return current_prompt[::-1]

    def read_csv(self, prefix, result, force_single_value_attributes):
        if logging.root.level <= logging.DEBUG:
            logger.debug(result)
        result = "\n".join(r for r in result.split("\n") if ";" in r)
        if not bool(set(prefix.strip().split(";")) & set(result.split("\n")[0].strip().split(";"))):
            result = prefix + result
        result = self.fix_csv(result)
        try:
            result =  pd.read_csv(StringIO(result), sep=DELIMITER, quoting=csv.QUOTE_NONE).fillna("")  # type: ignore
        except Exception as e:
            logger.warning(f"Could not read output CSV ({e}):\n\n{result}.", exc_info=True)
            return pd.read_csv(StringIO(prefix), sep=DELIMITER)  # type: ignore

        multi_value_attributes = [c for c in result.columns if c not in force_single_value_attributes]
        single_value_attributes = [c for c in result.columns if c in force_single_value_attributes]
        result[multi_value_attributes] = result[multi_value_attributes].applymap(  # type: ignore
            lambda x: [(y[:-2] if y.endswith(".0") and y[:-2].isnumeric() else y)
                       for y in str(x).split(MULTI_VALUE_SEPARATOR)] if str(x).strip(" -") else "")
        result[single_value_attributes] = result[single_value_attributes].applymap(lambda x: [x])  # type: ignore
        return result

    def fix_csv(self, csv):
        num_delimiter = csv.split("\n")[0].count(DELIMITER)
        rows = csv.split("\n")
        result = []
        for row in rows:
            this_row_num_delimiter = row.count(DELIMITER)
            if this_row_num_delimiter < num_delimiter:
                result += [row + DELIMITER * (num_delimiter - this_row_num_delimiter)]
            elif this_row_num_delimiter > num_delimiter:
                result += [DELIMITER.join(row.split(DELIMITER)[:num_delimiter + 1])]
            else:
                result += [row]
        return "\n".join(result)


@define
class LLaMAEngine(LLMEngine):
    ckpt_dir: str = field()
    tokenizer_path: str = field()
    max_seq_len: int = field(default=4096)
    max_batch_size = field(default=1)
    generator = field(init=False, default=None)
    model_parallel_size = field(init=False, default=1)

    def __attrs_post_init__(self):
        if str(self.ckpt_dir).endswith("safetensors"):
            self.name = "LLaMA-FT"
            self.model_parallel_size = 1
        elif "7b" in Path(self.ckpt_dir).name:
            self.name = "LLaMA"
            self.model_parallel_size = 1
        elif "13b" in Path(self.ckpt_dir).name:
            self.name = "LLaMA13B"
            self.model_parallel_size = 2
        elif "70B" in Path(self.ckpt_dir).name:
            self.name = "LLaMA70B"
            self.model_parallel_size = 8
        else:
            raise ValueError("Could not infer model size.")
        self.max_batch_size = max(1, int(os.environ.get("WORLD_SIZE", 0)) // self.model_parallel_size)

    def get_cache_file(self):
        cache_file = "_".join(Path(self.ckpt_dir).parts[1:]) + ".pkl"
        cache_file = self.cache_dir / cache_file
        return cache_file

    def max_prompt_length(self, x):
        return self.max_seq_len - LLaMAEngine.max_result_tokens

    def setup(self):
        if not self.ckpt_dir.endswith(".safetensors"):
            self.generator = Llama.build(
                ckpt_dir=self.ckpt_dir,
                tokenizer_path=self.tokenizer_path,
                max_seq_len=self.max_seq_len,
                max_batch_size=self.max_batch_size,
                model_parallel_size=self.model_parallel_size)
        else:
            from peft import AutoPeftModelForCausalLM  # type: ignore
            hf_generator = namedtuple("hf_generator", ["model", "tokenizer", "accelerator"])
            accelerator = Accelerator()
            self.generator = hf_generator(
                accelerator.prepare(AutoPeftModelForCausalLM.from_pretrained(
                    str(Path(self.ckpt_dir).parent),
                    low_cpu_mem_usage=True,
                )),
                AutoTokenizer.from_pretrained(str(Path(self.ckpt_dir).parent)),
                accelerator
            )
        return super().setup()

    def execute(self, model_input, attributes, identifying_attribute, force_single_value_attributes, mode: EngineMode):
        if identifying_attribute is not None and identifying_attribute not in attributes:
            attributes = [identifying_attribute] + attributes

        use_hf = self.ckpt_dir.endswith(".safetensors")
        adjust_func = self.adjust_prompt_hf if use_hf else self.adjust_prompt
        truncate_func = partial(self.truncate_prompt, get_num_tokens_func=self.get_num_tokens_hf) \
            if use_hf else self.truncate_prompt
        translate_func = self.translate_hf if use_hf else self.translate
        model_input.prompts.operations = [truncate_func, adjust_func]

        cache_key = (tuple(model_input.data.index.unique(0)), tuple(attributes), torch.cuda.device_count(),
                     self.max_seq_len, self.max_batch_size, model_input.num_samples, model_input.finetune_split_size)
        cached_result = self.check_cache(cache_key)

        raw_results, prefixes = cached_result if cached_result is not None else translate_func(model_input.prompts)
        self.update_cache(cache_key, cached_result is None, (raw_results, prefixes))

        results = []
        for i, (result, prefix) in enumerate(zip(raw_results, prefixes)):
            result = self.read_csv(prefix, result, force_single_value_attributes)
            result.index = np.ones(len(result), dtype=int) * i  # type: ignore
            results.append(result)
        return self.finalize(model_input, attributes, identifying_attribute, results)

    def translate(self, prompts):
        raw_results = []
        current_batch = []
        prefixes = []
        for prompt, prefix in prompts:
            current_batch.append(prompt)
            prefixes.append(prefix)

            if len(current_batch) == self.max_batch_size:
                raw_results += self.generator.chat_completion(  # type: ignore
                    current_batch,  # type: ignore
                    max_gen_len=LLaMAEngine.max_result_tokens,
                    temperature=0.0,
                    top_p=1,
                )
                current_batch = []

        if len(current_batch):
            raw_results += self.generator.chat_completion(  # type: ignore
                current_batch,  # type: ignore
                max_gen_len=LLaMAEngine.max_result_tokens,
                temperature=0.0,
                top_p=1,
            )

        raw_results = [r['generation']['content'] for r in raw_results]
        return raw_results, prefixes

    def translate_hf(self, prompts):
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dataset = MyPromptLoaderDataset(prompts, tokenizer=self.generator.tokenizer)
        dataloader = DataLoader(dataset, batch_size=int(self.max_batch_size / world_size),
                                shuffle=False, num_workers=3)
        dataloader = self.generator.accelerator.prepare(dataloader)  # type: ignore
        prefixes, raw_results, global_idxs = [], [], []

        with torch.inference_mode():
            for inputs in dataloader:
                input_length = inputs["input_ids"].shape[1]
                max_new_tokens = self.max_seq_len - input_length
                outputs = self.generator.model.generate(**{k: v for k, v in inputs.items()
                                                           if k not in ("prefix", "global_idx")},
                                                        max_new_tokens=max_new_tokens,
                                                        temperature=0.0001)
                outputs = pad(outputs, (0, self.max_seq_len - outputs.shape[1]), value=0)
                gathered = self.generator.accelerator.gather(outputs)  # type: ignore
                gathered_prefixes = self.generator.accelerator.gather(inputs["prefix"])  # type:ignore
                gathered_global_idx = self.generator.accelerator.gather(inputs["global_idx"])  # type:ignore

                new_raw_results = [
                    self.generator.tokenizer.decode(g[input_length:], skip_special_tokens=True).strip()  # type:ignore
                    for g in gathered
                ]
                new_prefixes = [
                    self.generator.tokenizer.decode(g, skip_special_tokens=True).strip()  # type: ignore
                    for g in gathered_prefixes
                ]
                new_global_idx = [g.item() for g in gathered_global_idx]
                raw_results.extend(new_raw_results)
                prefixes.extend(new_prefixes)
                global_idxs.extend(new_global_idx)
        order = np.unique(global_idxs, return_index=True)[1]
        raw_results = [raw_results[i] for i in order]
        prefixes = [prefixes[i] for i in order]
        return raw_results[:prompts.num_prompts], prefixes[:prompts.num_prompts]

    def adjust_prompt(self, prompt):
        return prompt[1:] if prompt[0]["role"] == "assistant" else prompt

    def adjust_prompt_hf(self, prompt):
        prompt[1:] if prompt[0]["role"] == "assistant" else prompt

        system_prompt = "You translate texts to tables."
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt[0]["content"].strip()} [/INST]"""

    def get_num_tokens(self, text):
        return len(self.generator.tokenizer.encode(text, True, True))

    def get_num_tokens_hf(self, text):
        return self.generator.tokenizer(text, return_tensors="pt")["input_ids"].shape[1]

    def get_model_name(self):
        return "llama"


class MyPromptLoaderDataset(IterableDataset):
    def __init__(self, prompt_loader, tokenizer):
        self.prompt_loader: PromptLoader = prompt_loader
        self.tokenizer = tokenizer

    @property
    def num_processes(self):
        return self.prompt_loader.num_processes

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        for i, (x, p) in enumerate(self.prompt_loader.iter_distributed(worker_info.num_workers, worker_info.id)):
            global_idx = i * worker_info.num_workers + worker_info.id
            yield {
                "prefix": self.tokenizer.encode(p, return_tensors="pt")[0],
                "global_idx": torch.tensor([global_idx]),
                **{k: v[0]
                    for k, v in self.tokenizer(x, return_tensors="pt", max_length=3072, padding="max_length",
                                               truncation=True).items()}
            }
