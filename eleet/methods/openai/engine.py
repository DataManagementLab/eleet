import logging
from sys import stderr, stdout

from attr import field
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import numpy as np
from attrs import define
from eleet.methods.base_engine import EngineMode
from eleet.methods.llama.engine import LLMEngine
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate


logger = logging.getLogger(__name__)


LLM_COST = {
    "gpt-4-0125-preview": (0.01, 0.03),
    "gpt-4-0613": (0.03, 0.06),
    "gpt-3.5-turbo-0125": (0.0005, 0.0015),
    "gpt-3.5-turbo-0613": (0.0015, 0.0020),
    "gpt-3.5-turbo-1106": (0.0010, 0.0020)
}

LLM_CONTEXT_LENGTH = {
    "gpt-4-0125-preview": 128000,
    "gpt-4-0613": 8192,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-1106": 16385
}

MODELS = {
    "gpt-4-0125-preview": ChatOpenAI(model_name="gpt-4-0125-preview", temperature=.0, max_tokens=LLMEngine.max_result_tokens),  # type: ignore
    "gpt-4-0613": ChatOpenAI(model_name="gpt-4-0613", temperature=.0, max_tokens=LLMEngine.max_result_tokens),  # type: ignore
    "gpt-3.5-turbo-0125": ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=.0, max_tokens=LLMEngine.max_result_tokens),  # type: ignore
    "gpt-3.5-turbo-0613": ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=.0, max_tokens=LLMEngine.max_result_tokens),  # type: ignore
    "gpt-3.5-turbo-1106": ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=.0, max_tokens=LLMEngine.max_result_tokens)  # type: ignore
}


@define
class OpenAIEngine(LLMEngine):
    llm: ChatOpenAI = field(converter=lambda x:  # type: ignore
                            MODELS.get(x, ChatOpenAI(model_name=x,  #type: ignore
                                       temperature=.0, max_tokens=LLMEngine.max_result_tokens)))
    name = field(init=False, default="GPT-3")
    parallel_requests: bool = field(default=False)
    only_cost_estimation: bool = field(default=False, init=False)
    cost_estimation_cache = set()

    def __attrs_post_init__(self):
        self.name = self.llm.model_name

    def max_prompt_length(self, x):
        max_prompt_length = {
            k: v - LLMEngine.max_result_tokens
            for k, v in LLM_CONTEXT_LENGTH.items()
        }
        if x.startswith("ft:"):
            x = "gpt-3.5-turbo-0613"
        return max_prompt_length[x]

    def get_cache_file(self):
        cache_file = f"{self.llm.model_name}.pkl"
        cache_file = self.cache_dir / cache_file
        return cache_file

    def execute(self, model_input, attributes, identifying_attribute, force_single_value_attributes, mode: EngineMode):
        if identifying_attribute is not None and identifying_attribute not in attributes:
            attributes = [identifying_attribute] + attributes
        model_input.prompts.operations = [self.truncate_prompt, self.adjust_prompt]

        cache_key = (tuple(model_input.data.index.unique(0)), tuple(attributes), self.parallel_requests,
                     model_input.num_samples, model_input.finetune_split_size)
        if self.only_cost_estimation:
            self.estimate_costs(model_input.prompts, cache_key)
            return self.get_empty_result(model_input.data, attributes)

        cached_result = self.check_cache(cache_key)
        raw_results, prefixes = cached_result if cached_result is not None else self.translate(model_input.prompts)
        self.update_cache(cache_key, cached_result is None, (raw_results, prefixes))

        results = []
        for i, (result, prefix) in enumerate(zip(raw_results, prefixes)):
            result = self.read_csv(prefix, result, force_single_value_attributes)
            result.index = np.ones(len(result), dtype=int) * i  # type: ignore
            results.append(result)
        return self.finalize(model_input, attributes, identifying_attribute, results)

    def translate(self, prompts):
        if self.parallel_requests:
            prompts, prefixes = zip(*prompts)
            chains = {str(i): (ChatPromptTemplate.from_messages(p) | self.llm) for i, p in enumerate(prompts)}
            raw_results = RunnableMap(chains).invoke({})
        else:
            raw_results = dict()
            prefixes = list()
            for i, (prompt, prefix) in enumerate(prompts):
                chain = ChatPromptTemplate.from_messages(prompt) | self.llm
                raw_results[str(i)] = chain.invoke({})
                prefixes.append(prefix)
        raw_results = sorted(raw_results.items(), key=lambda x: int(x[0]))
        raw_results = [(result[1] if isinstance(result[1], str) else result[1].content).strip()
                        for result in raw_results]
        return raw_results, prefixes

    def get_num_tokens(self, text):
        return self.llm.get_num_tokens(text) + 30  # buffer

    def get_model_name(self):
        return self.llm.model_name

    def adjust_prompt(self, prompt):
        prompt = [(AIMessage if m["role"] == "assistant" else HumanMessage)(content=m["content"])
                       for m in prompt]
        prompt = "\n\n".join(str(p.content) for p in prompt) if isinstance(self.llm, OpenAI) else prompt
        return prompt

    def estimate_costs(self, prompts, cache_key, limit=2**32):
        if cache_key in OpenAIEngine.cost_estimation_cache:
            logger.warning("Skipping cost estimation for this query. Will use cached result here.")
            return
        OpenAIEngine.cost_estimation_cache.add(cache_key)
        prompts = [p for _, (p, _) in zip(range(limit), prompts)]
        avg_num_messages = np.mean([len(p) for p in prompts])
        num_tokens_prompts = sum([self.llm.get_num_tokens(m.content) for p in prompts for m in p])

        pseudo_results = [p[1] for p in prompts]
        num_tokens_results = sum([self.llm.get_num_tokens(m.content) for m in pseudo_results])

        print(f"Estimated number of tokens: {num_tokens_prompts + num_tokens_results}, "
              f"{num_tokens_prompts} for prompts, {num_tokens_results} for results. "
              f"Number of prompts: {len(prompts)}. Average number of messages: {avg_num_messages}")
        cost = num_tokens_prompts * (LLM_COST[self.llm.model_name][0] / 1000) \
            + num_tokens_results * (LLM_COST[self.llm.model_name][1] / 1000)
        print(f"Estimated costs: {cost} $")
        stderr.flush()
        stdout.flush()

