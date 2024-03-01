import logging
import time
from attr import define
import openai
from eleet.methods.llama.finetuning import LLMFinetuningEngine
from eleet.methods.openai.engine import OpenAIEngine
from openai import OpenAI


logger = logging.getLogger(__name__)


@define
class OpenAIFinetuningEngine(OpenAIEngine, LLMFinetuningEngine):

    def __attrs_post_init__(self):
        LLMFinetuningEngine._post_init(self)
        OpenAIEngine.__attrs_post_init__(self)

    def _finetune_with_split_size(self, dataset_name, finetuning_inputs, valid_inputs, callbacks, split_size, **kwargs):
        with self.get_train_split(finetuning_inputs, valid_inputs, split_size) as train_split:
            logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.DEBUG)
            logger.info(f"Running training on split of size {train_split.name}.")
            client = OpenAI()
            file = client.files.create(
                file=(train_split / "train").open("rb"),
                purpose="fine-tune"
            )
            scheduled = False
            while not scheduled:
                try:
                    client.fine_tuning.jobs.create(
                        training_file=file.id,
                        model=self.name,
                        suffix=f"{dataset_name}--{split_size}"
                    )
                    scheduled = True
                except openai.RateLimitError:
                    print("Too many Finetuning jobs. Sleeping")
                    time.sleep(60)
            print(f"Started Finetuning Job for {dataset_name} and Split size {split_size}")
