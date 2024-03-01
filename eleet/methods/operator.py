from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import copy
from functools import partial, reduce
from attrs import define, field
from typing import List, Optional, Type, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from eleet.database import Table, TextCollectionLabels, TextCollectionTable, Database
from eleet.methods.base_engine import BaseEngine, EngineMode
from eleet.methods.base_preprocessor import BasePreprocessor
from eleet.methods.labels.engine import LabelEngine


def shorten_project_columns(project_columns):
    if len(project_columns) <= 4:
        return ",".join(project_columns)
    return ",".join(project_columns[:4]) + "," + project_columns[4][:2] + "…"

@define
class Operator(ABC):
    operands: List[Union[str, "Operator"]] = field()
    operand_types: List[Type] = field(init=False)

    def execute(self, database: Database, preprocessor: BasePreprocessor, engine: Optional[BaseEngine]):
        if engine is None and any(isinstance(o, MMOperator) for o in self.operands):
            raise RuntimeError("No two MMOps in one plan allowed for finetuning!")

        operands = [
            database.get(o) if isinstance(o, str) else o.execute(database, preprocessor, engine)
            for o in self.operands 
        ]
        assert len(operands) == len(self.operand_types)
        for operand, operand_type in zip(operands, self.operand_types):
            assert isinstance(operand, operand_type)
        result = self._execute(operands, preprocessor, engine, with_labels=isinstance(engine, LabelEngine))
        return result

    def get_finetuning_data(self, database: Database, preprocessor: BasePreprocessor, shuffle: bool):
        for o in self.operands:
            if isinstance(o, MMOperator):
                return o.get_finetuning_data(database, preprocessor, shuffle)
            
        for o in self.operands:
            r = o.get_finetuning_data(database, preprocessor, shuffle)
            if r is not None:
                return r

    @abstractmethod
    def _execute(self, operands, preprocessor, engine, with_labels=False):
        pass


@define
class Join(Operator):
    operand_types: int = field(init=False, default=[Table, Table])
    join_key: str = field()

    def _execute(self, operands, preprocessor, engine, additional_join_keys=(), with_labels=False):
        left = operands[0].data
        right = operands[1].data
        join_keys = [self.join_key] + list(additional_join_keys)
        left_additional_keys = [n for n in left.index.names if n not in join_keys]
        right_additional_keys = [n for n in right.index.names if n not in join_keys]
        if left_additional_keys:
            left = left.reset_index(left_additional_keys)
        if right_additional_keys:
            right = right.reset_index(right_additional_keys)
        result = right.merge(left, left_index=True, right_index=True)
        if left_additional_keys or right_additional_keys:
            result.set_index(left_additional_keys + right_additional_keys, inplace=True, append=True)
        if with_labels and any(o.labels is not None for o in operands):
            raise NotImplementedError("Computation of labels for Join after MMOp not implemented yet.")
        return Table(name=f"{operands[0].name} ⨝ {operands[1].name}", data=result, key_columns=result.index.names)


@define
class Selection(Operator):
    operand_types: int = field(init=False, default=[Table])
    selectivity: float = field()

    def _execute(self, operands, preprocessor, engine, with_labels=False):
        selected = operands[0].data.index.sort_values()[: max(int(len(operands[0].data.index) * self.selectivity), 1)]
        data = operands[0].data.loc[selected]
        return Table(name=f"σ_{self.selectivity}({operands[0].name})", data=data, key_columns=operands[0].key_columns)


@define
class Projection(Operator):
    operand_types: int = field(init=False, default=[Table])
    project_columns: List[str] = field()

    def _execute(self, operands, preprocessor, engine, with_labels=False):
        project_columns = [c for c in self.project_columns if c not in operands[0].key_columns]
        projected = operands[0].data[project_columns]
        name = f"π_{shorten_project_columns(self.project_columns)}({operands[0].name})"
        result = Table(name=name, data=projected, key_columns=operands[0].key_columns)
        return result



####################################

@define
class MMOperator(Operator):

    @contextmanager
    def get_finetuning_data(self, database: Database, preprocessor: BasePreprocessor, shuffle: bool):
        operands = [
            database.get(o) if isinstance(o, str) else o.execute(database, preprocessor, engine=None)
            for o in self.operands 
        ]
        assert len(operands) == len(self.operand_types)
        for operand, operand_type in zip(operands, self.operand_types):
            assert isinstance(operand, operand_type)
        with self._get_finetuning_data(operands, preprocessor, shuffle) as result:
            yield result


@define
class MMJoin(MMOperator, Join):
    operand_types: int = field(init=False, default=[Table, TextCollectionTable])
    join_key: str = field()
    project_columns: Optional[List[str]] = field(default=None)
    limit: Optional[int] = field(default=None)

    def _execute(self, operands, preprocessor, engine, with_labels=False):
        dummy_join = Join._execute(self, operands, preprocessor, engine)
        if with_labels:
            self._dummy_execute_labels(operands, preprocessor, engine, with_labels, dummy_join)
        project_columns = self.project_columns or \
            operands[0].attributes + [a for a in operands[1].attributes if a not in operands[0].attributes]
        extract_attributes = [c for c in project_columns 
                              if c in operands[1].attributes and c not in operands[0].attributes]
        if operands[1].delayed_selection and operands[1].delayed_selection.attribute not in extract_attributes:
            extract_attributes.append(operands[1].delayed_selection.attribute)
        with preprocessor.compute_model_input(
            data=dummy_join,
            report_table_name=operands[1].full_name,
            report_column=operands[1].text_column,
            extract_attributes=extract_attributes,
            identifying_attribute=operands[1].identifying_attribute,
            limit=self.limit,
            example_rows=None,
            multi_table=operands[1].multi_table
        ) as model_input:
            result = engine.execute(model_input, extract_attributes, operands[1].identifying_attribute,
                                    operands[1].force_single_value_attributes, mode=EngineMode.JOIN)
            if operands[1].delayed_selection:
                result = operands[1].delayed_selection.func(result)[project_columns]
            result = dummy_join.data.reset_index().join(result, how="inner", rsuffix="__tmp__")[project_columns] \
                .set_index([n for n in dummy_join.data.index.names if n in project_columns])
            result = result[[c for c in result.columns if not c.endswith("__tmp__")]]
            name = f"{operands[0].name} ⨝ {operands[1].name}"
            if self.project_columns is not None:
                name = f"π_{shorten_project_columns(self.project_columns)}({name})"
            return Table(name=name, data=result, key_columns=result.index.names)

    def _dummy_execute_labels(self, operands, preprocessor, engine, with_labels, dummy_join):
        normed_table = Table(name=f"normed {operands[1].name}",
                             data=operands[1].labels.normed,
                             key_columns=operands[1].labels.normed.index.names)
        alignment_table = Table(name=f"alignment {operands[1].name}",
                                data=operands[1].labels.alignments,
                                key_columns=operands[1].labels.normed.index.names)
        label_dummy_join_normed = Join._execute(
            self, [operands[0], normed_table], preprocessor, engine,
            additional_join_keys=[operands[1].identifying_attribute], with_labels=with_labels)
        label_dummy_join_alignments = Join._execute(
            self, [operands[0], alignment_table], preprocessor, engine,
            additional_join_keys=[operands[1].identifying_attribute], with_labels=with_labels)
        dummy_join.labels = TextCollectionLabels(
            normed=label_dummy_join_normed.data,
            alignments=label_dummy_join_alignments.data
        )

    @contextmanager
    def _get_finetuning_data(self, operands, preprocessor, shuffle):
        dummy_join = Join._execute(self, operands, preprocessor, engine=None)
        alignments = Join._execute(self, [operands[0], operands[1].alignments_table], preprocessor, engine=None,
                                   additional_join_keys=[operands[1].identifying_attribute])
        normed = Join._execute(self, [operands[0], operands[1].normed_table], preprocessor, engine=None,
                               additional_join_keys=[operands[1].identifying_attribute])
        project_columns = self.project_columns or \
            operands[0].attributes + [a for a in operands[1].attributes if a not in operands[0].attributes]
        extract_attributes = [c for c in project_columns 
                              if c in operands[1].attributes and c not in operands[0].attributes]
        with preprocessor.compute_finetuning_data(
            report_table_name=operands[1].full_name,
            data=dummy_join,
            report_column=operands[1].text_column,
            extract_attributes=extract_attributes,
            identifying_attribute=operands[1].identifying_attribute,
            example_rows=None,
            alignments=alignments,
            normed=normed,
            shuffle=shuffle,
            multi_table=operands[1].multi_table
        ) as finetuning_data:
            yield finetuning_data


@define
class MMUnion(MMOperator):
    operand_types: List = field(init=False, default=[Table, TextCollectionTable])
    only_extractions_in_result: bool = field(default=False)
    limit: Optional[int] = field(default=None)

    def _execute(self, operands, preprocessor, engine, with_labels=False):
        project_columns = [c for c in operands[0].attributes
                           if c not in operands[1].key_columns and c in operands[1].attributes]
        extract_attributes = copy(project_columns)
        if operands[1].delayed_selection and operands[1].delayed_selection.attribute not in extract_attributes:
            extract_attributes.append(operands[1].delayed_selection.attribute)
        with preprocessor.compute_model_input(
            data=operands[1],
            report_column=operands[1].text_column,
            report_table_name=operands[1].full_name,
            extract_attributes=extract_attributes,
            identifying_attribute=operands[1].identifying_attribute,
            example_rows=operands[0],
            multi_table=operands[1].multi_table,
            limit=self.limit
        ) as model_input:
            result = engine.execute(model_input, extract_attributes, operands[1].identifying_attribute,
                                    operands[1].force_single_value_attributes, mode=EngineMode.UNION)
            if operands[1].delayed_selection:
                result = operands[1].delayed_selection.func(result)[project_columns]
            index_map = {k: v for k, v in enumerate(operands[1].data.index)}
            result.index = pd.Index([index_map[i] for i in result.index], name=operands[1].data.index.name)
            if not self.only_extractions_in_result:
                result = pd.concat((operands[0].data, result))
            return Table(name=f"{operands[0].name} ∪ {operands[1].name}", data=result, key_columns=result.index.names)

    @contextmanager
    def _get_finetuning_data(self, operands, preprocessor, shuffle):
        extract_attributes = [c for c in operands[1].attributes if c not in operands[1].key_columns]
        with preprocessor.compute_finetuning_data(
            report_table_name=operands[1].full_name,
            data=operands[1],
            report_column=operands[1].text_column,
            extract_attributes=extract_attributes,
            identifying_attribute=operands[1].identifying_attribute,
            example_rows=operands[0],
            alignments=operands[1].alignments_table,
            normed=operands[1].normed_table,
            shuffle=shuffle,
            multi_table=operands[1].multi_table
        ) as finetuning_data:
            yield finetuning_data


@define
class MMScan(MMOperator):
    operand_types: int = field(init=False, default=[TextCollectionTable])
    project_columns: Optional[List[str]] = field(default=None)
    limit: Optional[int] = field(default=None)

    def _execute(self, operands, preprocessor, engine, with_labels=False):
        project_columns = self.project_columns or operands[0].attributes
        extract_attributes = [c for c in project_columns if c not in operands[0].key_columns]
        if operands[0].delayed_selection and operands[0].delayed_selection.attribute not in extract_attributes:
            extract_attributes.append(operands[0].delayed_selection.attribute)
        with preprocessor.compute_model_input(
            data=operands[0],
            report_column=operands[0].text_column,
            report_table_name=operands[0].full_name,
            extract_attributes=extract_attributes,
            identifying_attribute=operands[0].identifying_attribute,
            example_rows=None,
            multi_table=operands[0].multi_table,
            limit=self.limit
        ) as model_input:
            result = engine.execute(model_input, extract_attributes, operands[0].identifying_attribute,
                                    operands[0].force_single_value_attributes, mode=EngineMode.SCAN)

            if operands[0].delayed_selection:
                result = operands[0].delayed_selection.func(result)[project_columns]
            index_map = {k: v for k, v in enumerate(operands[0].data.index)}
            result.index = pd.Index([index_map[i] for i in result.index], name=operands[0].data.index.name)
            result = result[[p for p in project_columns if p not in result.index.names]]
            name = f"π_{shorten_project_columns(project_columns)}({operands[0].name})"
            result = Table(name=name, data=result, key_columns=result.index.names)
            if with_labels:
                self._scan_with_labels(operands, project_columns, result)
            return result

    def _scan_with_labels(self, operands, project_columns, result):
        project_columns = [c for c in project_columns if c in operands[0].labels.normed.columns]
        if operands[0].delayed_selection:
            normed = operands[0].delayed_selection.func(operands[0].labels.normed.loc[:self.limit])[project_columns]
            alignments = operands[0].labels.alignments.loc[normed.index, project_columns]
        else:
            normed = operands[0].labels.normed.loc[:self.limit, project_columns]
            alignments = operands[0].labels.alignments.loc[:self.limit, project_columns]
        labels = TextCollectionLabels(normed=normed, alignments=alignments)
        result.labels = labels


@define
class MMAggregation(MMOperator):
    operand_types: int = field(init=False, default=[Table])
    attribute: str = field()

    def _execute(self, operands, preprocessor, engine, with_labels=False):
        data = operands[0].data.applymap(lambda x: [] if x == "" else x)
        col_values = data[self.attribute].reset_index().reset_index().explode(self.attribute)
        col_values = col_values.loc[~col_values[self.attribute].isna()]
        col_values.columns = ["index", "orig_index", self.attribute]

        if isinstance(engine, LabelEngine):
            groups = engine.aggregate(col_values[["orig_index", self.attribute]],
                                      operands[0].labels.normed[self.attribute])
        else:
            groups = engine.aggregate(col_values[self.attribute])

        data[self.attribute] = float("nan")
        for ids, name in groups.items():
            values = [name] * len(ids)
            data.iloc[col_values.iloc[list(ids)]["index"].tolist(),
                      data.columns.tolist().index(self.attribute)] = pd.Series(values)

        grouped = data.groupby(self.attribute).agg(partial(reduce, lambda a, b: a + b))
        grouped = grouped.loc[[x for x in grouped.index if len(x) and x[0] not in ("#", ".", ".")]]
        return Table(name=f"G_{self.attribute}({operands[0].name})", data=grouped, key_columns=[self.attribute])


DelayedSelection = namedtuple("DelayedSelection", ["attribute", "func"])
@define
class MMSelection(MMOperator):
    operand_types: List[Type] = field(init=False, default=[TextCollectionTable])
    attribute: str = field()
    selectivity: float = field()
    limit: Optional[int] = field(default=None)

    def _execute(self, operands, preprocessor, engine, with_labels=False):
        value, selectivity = self._choose_condition(operands, use_normed=engine.use_normed_conditions)
        name = f"σ_{self.attribute}={value}[{selectivity:.3f}]({operands[0].name})"
        if engine.use_normed_conditions:
            return self.baseline_select(operands, value, name)

        id_attr = operands[0].identifying_attribute
        extract_attributes = [self.attribute] if self.attribute == id_attr or id_attr is None \
            else [id_attr, self.attribute]
        with preprocessor.compute_model_input(
            data=operands[0],
            report_column=operands[0].text_column,
            report_table_name=operands[0].full_name,
            extract_attributes=extract_attributes,
            identifying_attribute=id_attr,
            example_rows=None,
            multi_table=operands[0].multi_table,
            limit=self.limit,
            index_mode=True
        ) as model_input:
            result_index, result_synonyms = \
                engine.build_index(extract_attributes, operands, model_input)

            selected_ids = sorted(result_index[value])

            result_data = operands[0].data.loc[selected_ids]
            result_table = copy(operands[0])
            result_text_collection = copy(operands[0].text_collection)
            result_table.name = name
            result_table.text_collection = result_text_collection

            if selected_ids:
                selected_synonyms = result_synonyms[value]
                result_table.delayed_selection = DelayedSelection(self.attribute, partial(self._select,
                                                                                          value=selected_synonyms))
            result_text_collection.data = result_data
            result_text_collection.text_tables = {k: (v if k != operands[0].name else result_table)
                                                  for k, v in result_text_collection.text_tables.items()}
            return result_table


    def baseline_select(self, operands, value, name):
        result = copy(operands[0])
        result.name = name
        result.delayed_selection = DelayedSelection(self.attribute, partial(self._select, value=value))
        return result

    def _choose_condition(self, operands, use_normed=True):
        combined = operands[0].data.merge(
            pd.concat((operands[0].labels.normed[self.attribute],
                       operands[0].labels.alignments[self.attribute]), axis=1), 
            on=operands[0].key_columns)
        combined.columns = ["text", "normed", "alignments"]
        combined["surface"] = combined.apply(lambda x: [x["text"][a[0]: a[1]] for a in x["alignments"]], axis=1)
        label_values = combined.apply(
            lambda x: [(a.lower(), b.lower()) for a, b in zip(x["normed"], x["surface"])],
            axis=1).explode()
        label_values = label_values[~label_values.isna()].apply(lambda x: pd.Series(x, index=["normed", "surface"]))
        _, index, counts = np.unique(label_values["normed"], return_counts=True, return_index=True)
        selectivities = counts / len(operands[0].data)
        chosen_id = np.abs(selectivities - self.selectivity).argmin()
        value = label_values.iloc[index[chosen_id]]["normed" if use_normed else "surface"]
        selectivity = selectivities[chosen_id]
        return value, selectivity

    def _select(self, data, value):
        selected = data[self.attribute].apply(partial(self._eval_select_condition, target_value=value))
        return data.loc[selected]

    def _eval_select_condition(self, value, target_value):
        for v in value:
            if isinstance(v, str):
                v = v.lower()
            if isinstance(v, set) and target_value in v:
                return True
            if isinstance(v, tuple) and isinstance(target_value, set) and v[0] in target_value:
                return True
            if not isinstance(v, (set, tuple)) and target_value == v:
                return True
        return False
