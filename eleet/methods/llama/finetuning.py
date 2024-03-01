import argparse
from contextlib import contextmanager
from datetime import datetime
import logging
from pathlib import Path
from attr import define
import numpy as np
import json
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict

from eleet.methods.llama.engine import LLaMAEngine


logger = logging.getLogger(__name__)
HF_TOKEN = os.environ.get("HF_TOKEN")


class LLMFinetuningEngine():
    finetuned_model_local_folder_name = None

    def _post_init(self):
        self.args = self.parse_args()
        self.rng = np.random.default_rng()

    def finetune(self, dataset_name, finetuning_inputs, valid_inputs=None, callbacks=[]):
        max_split_size = self.get_max_split_size(finetuning_inputs)
        finetune_split_sizes = sorted(set(min(x, max_split_size) for x in self.args.finetune_split_sizes), reverse=True)
        this_dir = None
        if self.finetuned_model_local_folder_name is not None:
            now = datetime.now()
            date_prefix = now.strftime("%Y-%m-%d_%H-%M-%S")
            this_dir = Path(f"models/{dataset_name}/{self.finetuned_model_local_folder_name}/{date_prefix}").absolute()
            this_dir.mkdir(exist_ok=True, parents=True)
        for split_size in map(int, finetune_split_sizes):
            self._finetune_with_split_size(dataset_name, finetuning_inputs, valid_inputs, callbacks,
                                           split_size=split_size, this_dir=this_dir)

    def get_max_split_size(self, finetuning_inputs):
        max_split_size = []
        for p in Path(finetuning_inputs).iterdir():
            if not p.name.startswith("train"):
                continue
            with p.open() as f:
                max_split_size += [len(f.readlines())]
        return max(max_split_size)

    def _finetune_with_split_size(self, dataset_name, finetuning_inputs, valid_inputs, callbacks, split_size, this_dir):
        raise NotImplementedError

    @contextmanager
    def get_train_split(self, finetuning_inputs, valid_inputs, split_size):
        split_dir = Path(finetuning_inputs) / str(split_size)
        split_dir.mkdir()
        for file in Path(finetuning_inputs).iterdir():
            if file.is_dir():
                continue
            with file.open() as f_in:
                with (split_dir / file.name.split("---")[0]).open("a") as f_out:
                    for i, line in enumerate(f_in):
                        if i >= split_size:
                            break
                        print(line, file=f_out, end="")

        for file in Path(valid_inputs).iterdir():
            with file.open() as f_in:
                with (split_dir / file.name.split("---")[0]).open("a") as f_out1:
                    with (split_dir / file.name.replace("valid", "test").split("---")[0]).open("a") as f_out2:
                        for line in f_in:
                            print(line, file=f_out1, end="")
                            print(line, file=f_out2, end="")
        yield split_dir.absolute()

    def parse_args(self):
        """Add the arguments of the parser."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--finetune-split-sizes", type=int, nargs="+", default=[(2 ** i) for i in range(7, 15)])
        args = parser.parse_args()
        return args


@define
class LLamaFinetuningEngine(LLaMAEngine, LLMFinetuningEngine):
    finetuned_model_local_folder_name = "llama"

    def __attrs_post_init__(self):
        LLMFinetuningEngine._post_init(self)
        LLaMAEngine.__attrs_post_init__(self)

    def _finetune_with_split_size(self, dataset_name, finetuning_inputs, valid_inputs, callbacks, split_size, this_dir):
        with self.get_train_split(finetuning_inputs, valid_inputs, split_size) as train_split:

            result_dir = this_dir / f"checkpoint.{split_size}"
            result_dir.mkdir(exist_ok=True, parents=True)
            with (result_dir / "args.json").open("w") as f:
                json.dump(vars(self.args), f)

            self.run_finetuning(
                    output_dir=str(result_dir),
                    input_dir=str(train_split),
            )
            print("done training llama")

    def parse_args(self):
        """Add the arguments of the parser."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--finetune-split-sizes", type=int, nargs="+", default=[(2 ** i) for i in range(7, 15)])
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--per-device-train-batch-size", type=int)
        parser.add_argument("--gradient-accumulation-steps", type=int)
        parser.add_argument("--quantization-num-bits", type=int)
        parser.add_argument("--lora-r", type=int, default=16)
        parser.add_argument("--lora-alpha", type=int, default=32)
        parser.add_argument("--lora-dropout", type=float, default=0.05)
        parser.add_argument("--max-steps", type=int, default=250)
        parser.add_argument("--max-epochs", type=int, default=6)
        args = parser.parse_args()
        return args

    def load_dataset(self, path):
        datasets = {}
        for split in ("train", "valid", "test"):
            user_msgs = []
            ai_msgs = []
            with open(Path(path) / split, "r") as f:
                for line in f:
                    line = json.loads(line)
                    messages = line["messages"]
                    user_msgs.append(messages[0]["content"])
                    ai_msgs.append(messages[1]["content"])
            datasets[split] = Dataset.from_dict({"user": user_msgs, "ai": ai_msgs})
        return DatasetDict(datasets)


    def generate_training_prompt(
        self,
        user: str, ai: str
    ) -> str:
        system_prompt = "You translate texts to tables."
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{ user.strip() } [/INST] { ai.strip() } </s>
"""

    def generate_text(self, data_point):
        user = data_point["user"]
        ai = data_point["ai"]
        return {
            "user": user,
            "ai": ai,
            "text": self.generate_training_prompt(user, ai),
        }


    def process_dataset(self, data: Dataset):
        return (
            data.shuffle(seed=42)
            .map(self.generate_text)
        )


    def create_model_and_tokenizer(self, model_name):
        if self.args.quantization_num_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
            )
        elif self.args.quantization_num_bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError("Invalid quantization_num_bits")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            token=HF_TOKEN
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = {a: b for b, a in tokenizer.vocab.items()}[3]
        tokenizer.padding_side = "right"

        return model, tokenizer


    def run_finetuning(self,
                       output_dir,
                       input_dir,
                       model_name="meta-llama/Llama-2-7b-chat-hf"):
        from peft import LoraConfig
        from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

        # DATASET
        dataset = self.load_dataset(input_dir)
        dataset["train"] = self.process_dataset(dataset["train"])
        dataset["valid"] = self.process_dataset(dataset["valid"])

        # Create Model and Tokenizer
        model, tokenizer = self.create_model_and_tokenizer(model_name)
        model.config.use_cache = False

        # LORA
        lora_r = self.args.lora_r
        lora_alpha = self.args.lora_alpha
        lora_dropout = self.args.lora_dropout
        lora_target_modules = [
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Data Collator
        response_template = "[/INST]"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, tokenizer=tokenizer,
            mlm=False
        )

        max_steps = min(
            (len(dataset["train"]) * self.args.max_epochs)
            // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps),
            self.args.max_steps
        )
        # TRAINING
        training_arguments = TrainingArguments(
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            logging_steps=1,
            learning_rate=self.args.learning_rate,
            fp16=True,
            max_grad_norm=0.3,
            # num_train_epochs=2,
            max_steps=max_steps,
            evaluation_strategy="steps",
            weight_decay=0.001,
            eval_steps=0.2,
            warmup_ratio=0.05,
            save_strategy="no",
            group_by_length=False,
            output_dir=output_dir,
            report_to=["tensorboard"],
            save_safetensors=True,
            lr_scheduler_type="cosine",
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            peft_config=peft_config,  # type: ignore
            dataset_text_field="text",
            max_seq_length=4096,
            tokenizer=tokenizer,
            args=training_arguments,
            data_collator=collator,
            packing=False
        )

        trainer.train()
        trainer.save_model()

