import os
from contextlib import ExitStack
from datetime import datetime, timedelta
from attr import Factory
from attrs import define, field
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast
from eleet.methods.base_engine import BaseEngine


@define
class Database():
    name: str = field()
    tables: Dict[str, "Table"] = field(converter=lambda x: {t.name: t for t in x})
    texts: Dict[str, "TextCollection"] = field(converter=lambda x: {t.name: t for t in x})

    def get(self, name):
        if "." in name:
            text_collection_name, table_name = name.split(".")
            return self.texts[text_collection_name].text_tables[table_name]
        return self.tables.get(name, self.texts.get(name, None))

    def names(self):
        return list(self.tables) + list(self.texts)

    def preprocess(self, preprocessor):
        pass

    def to_finetuning_dataset(self, query_plan, preprocessor):
        pass

    def execute_query(self, query_plan, preprocessor, engine: BaseEngine, measure_runtime: bool=False):
        engine.setup()
        start_time = datetime.now()
        result = query_plan.execute(self, preprocessor, engine)
        end_time = datetime.now()
        engine.shutdown()
        if measure_runtime:
            runtime = (end_time - start_time) + engine.runtime_correction
            print("**********************************************Final runtime", runtime, runtime - engine.index_build_time)
            result =  (result, runtime, engine.index_build_time)
        engine.index_build_time = timedelta()
        engine.runtime_correction = timedelta()
        return result

    def get_text_metadata(self, query_plan):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        def get_num_tokens(text):
            return len(tokenizer(text)["input_ids"])

        text_collection_table = query_plan.get_leaf_operands(self, TextCollectionTable)[0]
        num_words = text_collection_table.text_collection.data.apply(lambda x: len(x[-1].split()), axis=1)
        num_tokens = text_collection_table.text_collection.data.apply(lambda x: get_num_tokens(x[-1]), axis=1)
        metadata = pd.DataFrame({
            "num_words": num_words,
            "num_tokens": num_tokens
        })
        return metadata

    def finetune(self, query_plans, preprocessor, engine, db_valid: Optional["Database"]):
        with ExitStack() as stack:
            if not preprocessor.finetuning_independent_of_operator:
                finetuning_data = [stack.enter_context(p.get_finetuning_data(self, preprocessor, shuffle=True))
                                   for p in query_plans]
                valid_data = None
                if db_valid is not None:
                    valid_data = [stack.enter_context(p.get_finetuning_data(db_valid, preprocessor, shuffle=False))
                                  for p in query_plans]
            else:
                operands = self.find_leaf_operands_in_query_plans(query_plans, filter_type=TextCollectionTable)
                finetuning_data = stack.enter_context(preprocessor.compute_finetuning_data(
                    data=[self.get(o) for o in operands],
                    split="train",
                ))
                valid_data = None
                if db_valid is not None:
                    operands = db_valid.find_leaf_operands_in_query_plans(query_plans, filter_type=TextCollectionTable)
                    valid_data = stack.enter_context(preprocessor.compute_finetuning_data(
                        data=[db_valid.get(o) for o in operands],
                        split="valid"
                    ))
            engine.finetune(dataset_name=self.name, finetuning_inputs=finetuning_data, valid_inputs=valid_data)

    def find_leaf_operands_in_query_plans(self, query_plans, filter_type=object):
        from eleet.methods.operator import Operator
        if isinstance(query_plans, (list, tuple)):
            return {y for x in query_plans
                    for y in self.find_leaf_operands_in_query_plans(x, filter_type=filter_type)}
        if isinstance(query_plans, Operator):
            return {y for x in query_plans.operands
                    for y in self.find_leaf_operands_in_query_plans(x, filter_type=filter_type)}
        if isinstance(query_plans, str):
            if isinstance(self.get(query_plans), filter_type):
                return {query_plans}
            return {}
        raise ValueError(query_plans)

    def build_index(self, table, column, preprocessor, engine):
        pass

    def build_materialized_view(self, query_plan, preprocessor, engine):
        pass

    def serialize(self):
        pass


@define
class Table():
    name: str = field()
    data: pd.DataFrame = field()
    key_columns: List[str] = field()
    preprocessed: Optional[Dict] = field(init=False, default=None)
    labels: Optional["TextCollectionLabels"] = field(init=False, default=None)

    def __attrs_post_init__(self):
        if self.data.index.names != self.key_columns:
            self.data.set_index(self.key_columns, inplace=True)

    def preprocess(self):
        pass

    @property
    def attributes(self):
        return self.key_columns + self.data.columns.tolist()

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data.applymap(str))


@define
class TextCollectionTable():
    text_collection: "TextCollection" = field()
    name: str = field()
    attributes: List[str] = field()
    multi_row: bool = field()
    labels: Optional["TextCollectionLabels"] = field()
    identifying_attribute: Optional[str] = field()
    force_single_value_attributes: List[str] = field(converter=lambda x: set(x) if x is not None else set())
    simulate_single_table: bool = field(default=False)
    delayed_selection: Optional[bool] = field(init=False, default=None)

    @multi_row.validator
    def check_identifying_attribute(self, attribute, value):
        if value:
            assert self.identifying_attribute is not None
        else:
            assert self.identifying_attribute is None

    @property
    def multi_table(self):
        if self.simulate_single_table:
            return False
        return len(self.text_collection.text_tables) > 1

    @property
    def data(self):
        return self.text_collection.data

    @property
    def key_columns(self):
        return self.text_collection.key_columns

    @property
    def text_column(self):
        return self.text_collection.text_column
    
    @property
    def full_name(self):
        return (self.text_collection.name, self.name.split("(")[-1].split(")")[0])

    def __len__(self):
        return len(self.data)

    @property
    def normed_table(self):
        return Table(
            self.name,
            self.labels.normed,
            key_columns=self.key_columns + ([self.identifying_attribute] if self.identifying_attribute else [])
        )

    @property
    def alignments_table(self):
        return Table(
            self.name,
            self.labels.alignments,
            key_columns=self.key_columns + ([self.identifying_attribute] if self.identifying_attribute else [])
        )



@define
class TextCollection():
    name: str = field()
    data: pd.DataFrame = field()
    key_columns: List[str] = field()
    preprocessed: Optional[Dict] = field(init=False, default=None)
    text_tables: Dict[str, TextCollectionTable] = field(init=False, default=Factory(dict))

    @property
    def text_column(self):
        return self.data.columns[0]

    def setup_text_table(self, table_name: str, attributes: List[str],  multi_row: bool,
                         identifying_attribute=None, labels=None, force_single_value_attributes: Set = None,
                         simulate_single_table=False):
        attributes = [a for a in attributes if a not in self.key_columns]
        result = TextCollectionTable(text_collection=self, name=table_name, attributes=attributes,
                                     multi_row=multi_row, identifying_attribute=identifying_attribute, labels=labels,
                                     force_single_value_attributes=force_single_value_attributes,
                                     simulate_single_table=simulate_single_table)
        self.text_tables[table_name] = result
        return result

    def __attrs_post_init__(self):
        if self.data.index.names != self.key_columns:
            self.data.set_index(self.key_columns, inplace=True)

    @data.validator
    def check_data(self, attribute, value):
        if self.data.index.names != self.key_columns:
            assert len(value.columns) == 2
            assert value[value.columns[0]].dtype == int
            assert value[value.columns[1]].dtype == np.dtype("O")

    def preprocess(self):
        pass

    def __str__(self):
        return str(self.data)


@define
class TextCollectionLabels():
    normed: pd.DataFrame = field()
    alignments: pd.DataFrame = field()  # where in the text are the values

    def preprocess(self):
        pass
