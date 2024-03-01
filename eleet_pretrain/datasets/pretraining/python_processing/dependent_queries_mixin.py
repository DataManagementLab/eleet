import logging
from copy import copy, deepcopy
from functools import partial

from eleet_pretrain.datasets.pretraining.python_processing.utils import LABEL_COLUMN, NUM_COLS, group
from eleet_pretrain.datasets.pretraining.python_processing.utils import Query, TableRow, TableCell, TableHeader, QueryAnswer, QueryDependency
from eleet_pretrain.datasets.pretraining.python_processing.utils import shorten_uri


logger = logging.getLogger(__name__)


class DQ_MODE():
    PRETRAIN = 0
    EVAL_TRAIN_JOIN = 1
    EVAL_TABLE_DECODING_JOIN = 2
    EVAL_TRAIN_MULTI_UNION = 3
    EVAL_TABLE_DECODING_MULTI_UNION = 4

JOIN_MODES = (DQ_MODE.PRETRAIN, DQ_MODE.EVAL_TRAIN_JOIN, DQ_MODE.EVAL_TABLE_DECODING_JOIN)
UNION_MODES = (DQ_MODE.EVAL_TRAIN_MULTI_UNION, DQ_MODE.EVAL_TABLE_DECODING_MULTI_UNION)

DQ_TABLE_NAME_SUFFIX = "dependent_queries"
MU_TABLE_NAME_SUFFIX = "multi-union"


class DependentQueriesMixin():
    def compute_dependent_queries(self, header, rows, queries, mode):
        if mode in JOIN_MODES:
            yield from self.compute_dependent_queries_join_pretrain(header, rows, queries, mode)
        elif mode in UNION_MODES:
            yield from self.compute_dependent_queries_multi_union(header, rows, queries, mode)
        else:
            raise NotImplementedError

    # UNION ############################################################################################################

    def compute_dependent_queries_multi_union(self, header, rows, queries, mode):
        master_queries = [q for q in queries if q.col_id == "id"]
        if len([q for q in master_queries if q.answers]) != 2:
            missing = [q.row_id for q in master_queries if not q.answers]
            if missing:
                logger.warn(f"Skipping Multi Union Query, because Row Identifiers of {missing} not tagged in text.")
            else:
                logger.warn("This is not a multi-union (Couldn't find enough rows). Skipping! Query-Row: "
                            f"{set([q.row_id for q in queries])}")
            return
        queried_row_ids = sorted(set(q.row_id for q in queries))
        queried_rows = [r for r in rows if r.id in queried_row_ids]
        non_queried_rows = [r for r in rows if r.id not in queried_row_ids]

        merged_master_query = Query(queried_rows[0].text_id, "id",
                                    answers=[a for q in master_queries for a in q.answers])

        funcs = [self.multi_union_base_and_dependent_queries]
        if mode == DQ_MODE.EVAL_TRAIN_MULTI_UNION:
            funcs += [self.multi_union_queries_for_fixed_row_label]
        
        for f in funcs:
            for new_row, new_queries, table_suffix in f(queries=queries, master_query=merged_master_query,
                                                         queried_rows=queried_rows):
                yield table_suffix, header, non_queried_rows + [new_row], new_queries

    def multi_union_queries_for_fixed_row_label(self, queries, master_query, queried_rows):
        """For each possible answer of the master query, create a new sample with merged, non-dependent queries."""
        for i, (row_label, answers) in enumerate(QueryAnswer.group_by_surfaceform(master_query.answers).items()):
            row_id = self.get_dependent_row_id(master_query, row_label)
            covers_entities = set([shorten_uri(a.uri) for a in answers])
            if len(covers_entities) > 1:
                logger.warn(f"{row_label} is ambiguous in multi-union.")
                continue
            row_entity_id = next(iter(covers_entities))
            new_row = copy(next(iter(r for r in queried_rows if r.id==row_entity_id)))
            new_row.id = row_id
            relevant_queries = [q for q in queries if q.row_id == row_entity_id]
            new_queries = [Query(row_id=row_id, col_id=q.col_id, answers=q.answers, dependency=None)
                           for q in relevant_queries if q.col_id != "id"]
            if not any(q.answers for q in new_queries):
                continue
            table_suffix = f"{MU_TABLE_NAME_SUFFIX}-{i}"
            yield new_row, new_queries, table_suffix

    def multi_union_base_and_dependent_queries(self, queries, master_query, queried_rows):
        """Merges the queried rows and create merged queries depending on the given master query."""
        master_cell = TableCell(
            column_id="id",
            value=QueryAnswer.get_cell_value(master_query.answers),
            positions=QueryAnswer.get_boundaries(master_query.answers),
            num_values=len(master_query.answers)
        )
        row_id = self.get_dependent_row_id(master_query, None)
        master_query = Query(row_id, master_query.col_id, master_query.answers, dependency=None)
        merged_cells = [
                master_cell if c.column_id == "id"
                else TableCell(c.column_id, "[???]", [], 0)
                for c in queried_rows[0].cells]
        merged_row = TableRow(
                id=row_id, text_id=queried_rows[0].text_id, text=queried_rows[0].text, cells=merged_cells,
                relevant_text_boundaries=(min(r.relevant_text_boundaries[0] for r in queried_rows),
                                          max(r.relevant_text_boundaries[1] for r in queried_rows)),
            )

        merged_queries = []
        for _, answers in QueryAnswer.group_by_surfaceform(master_query.answers).items():
            covers_entities = set([shorten_uri(a.uri) for a in answers])
            relevant_queries = [q for q in queries if q.row_id in covers_entities and q.col_id != "id"]
            for q in relevant_queries:
                merged_queries.append(Query(
                        row_id=row_id, col_id=q.col_id, answers=q.answers,
                        dependency=QueryDependency(master_query, QueryAnswer.select_first_answer(answers))
                    ))
        new_queries = [master_query] + merged_queries
        table_suffix = f"{MU_TABLE_NAME_SUFFIX}-base"
        yield merged_row, new_queries, table_suffix


    # JOINS ############################################################################################################

    def compute_dependent_queries_join_pretrain(self, header, rows, queries, mode):
        """Compute queries that are dependent.
        
        E.g. Person won prize A in 1920 and prize B in 1923.
        An SQL query could SELECT the prize as well as the date.
        The date of winning the prize is dependent to which prize was won.
        """
        row_mapping = TableRow.get_mapping(rows)
        query_mapping = Query.group_by_column(queries)
        if "id" in query_mapping:
            del query_mapping["id"]

        for q_col_id, col_queries in query_mapping.items():
            new_columns = self.get_new_columns(col_queries)

            if not new_columns:
                continue

            new_header = self.get_dependent_columns(new_columns, header, q_col_id)
            if mode == DQ_MODE.PRETRAIN:
                funcs = [partial(self.join_dependent_queries, mask_entire_column=False)]
            elif mode == DQ_MODE.EVAL_TRAIN_JOIN:
                funcs = [self.join_dependent_queries, self.join_base_query]
            elif mode == DQ_MODE.EVAL_TABLE_DECODING_JOIN:
                funcs = [self.join_base_query]
            else:
                raise ValueError(mode)

            for f in funcs:
                yield from f(col_queries, header, new_header, row_mapping)

    def get_new_columns(self, col_queries):
        new_columns = set()  # columns given by dependent queries
        for q in col_queries:
            for answer in q.answers:
                for dependent_field in answer.dependent_query_data:
                    composite_col_id = (q.col_id, dependent_field['_id']['_attr_uri'].split('/')[-1])
                    new_columns.add(composite_col_id)
        return new_columns

    def join_dependent_queries(self, col_queries, old_header, new_header, row_mapping,
                                        mask_entire_column=True):
        new_rows, new_queries = list(), list()
        for q in col_queries:
            nq = self.get_join_dependent_queries(new_header, q, mask_entire_column=mask_entire_column)
            nr = self.get_dependent_rows(old_header, new_header, q, row_mapping[q.row_id], nq)
            new_rows.extend(nr)
            new_queries.extend(nq)
        for i, (new_rows_group, new_queries_group) in enumerate(self.group_dependent_query_rows(new_rows, new_queries)):
            table_name_suffix = "-".join((DQ_TABLE_NAME_SUFFIX, col_queries[0].col_id, str(i)))
            yield table_name_suffix, new_header, new_rows_group, new_queries_group

    def join_base_query(self, col_queries, old_header, new_header, row_mapping):
        new_queries, new_rows = list(), list()
        for q in col_queries:
            nq = self.get_join_base_query(new_header, q)
            nr = self.to_dependent_row(old_header, new_header, q, row_mapping[q.row_id], nq)
            new_queries.extend(nq)
            new_rows.append(nr)
        table_name_suffix = "-".join((DQ_TABLE_NAME_SUFFIX, col_queries[0].col_id))
        yield table_name_suffix, new_header, new_rows, new_queries

    def get_dependent_columns(self, new_columns, header, col_id):
        column_labels_dict = header.to_dict()
        new_header = (["id"] if "id" in header.column_ids else []) + [col_id]
        new_header += sorted(new_columns)
        new_header += [x for x in header.column_ids if x not in ("id",  col_id)]
        new_header = new_header[:(NUM_COLS + LABEL_COLUMN)]
        column_names = []
        for c in new_header:
            if isinstance(c, tuple):
                random_label = self.rng.choice([self.labels.get(c[1], c[1])] + self.aliases.get(c[1], []))
                column_names.append(column_labels_dict[c[0]] + " - " + random_label)
            else:
                column_names.append(column_labels_dict[c])
        # TODO fix stats
        return TableHeader(column_names=column_names, column_ids=new_header)

    def get_dependent_rows(self, old_header, new_header, query, row, dependent_queries):
        dependent_queries = Query.group_by_row(dependent_queries)
        result_rows = list()
        old_cells = dict(zip(old_header.column_ids, row.cells))
        for row_label, answers in QueryAnswer.group_by_surfaceform(query.answers).items():
            row_id = self.get_dependent_row_id(query, row_label)
            if row_id not in dependent_queries:
                continue
            dependent_queries_for_row = Query.group_by_column(dependent_queries[row_id], unique=True)
            new_cells = []
            for col_id in new_header.column_ids:
                if isinstance(col_id, tuple) and col_id in dependent_queries_for_row:
                    new_query = dependent_queries_for_row[col_id]
                    new_cells.append(TableCell(
                        column_id=col_id, 
                        value=QueryAnswer.get_cell_value(new_query.answers),
                        positions=QueryAnswer.get_boundaries(new_query.answers),
                        num_values=len(new_query.answers))
                    )
                elif isinstance(col_id, tuple):
                    new_cells.append(TableCell(column_id=col_id, value="None", positions=[], num_values=0))
                elif col_id == query.col_id:
                    new_cells.append(TableCell(col_id, row_label, QueryAnswer.get_boundaries(answers), 1))
                else:
                    new_cells.append(old_cells[col_id])
            result_rows.append(TableRow(row_id, new_cells, row.text_id, row.text, row.relevant_text_boundaries))
        return result_rows

    def to_dependent_row(self, old_header, new_header, query, row, dependent_queries):
        row_id = self.get_dependent_row_id(query, None)
        dependent_query_target_cells = set((q.row_id, q.col_id) for q in dependent_queries)
        old_cells = dict(zip(old_header.column_ids, row.cells))
        new_cells = []
        for col_id in new_header.column_ids:
            if isinstance(col_id, tuple):
                new_cells.append(TableCell(
                    column_id=col_id, 
                    value="[???]",  # will get masked
                    positions=[],
                    num_values=0)
                )
                if (row_id, col_id) not in dependent_query_target_cells:  # only for missing cells
                    dependent_queries.append(Query(row_id, col_id, answers=[], dependency=None))
            elif col_id == query.col_id:
                new_cells.append(TableCell(
                    column_id=col_id, 
                    value=QueryAnswer.get_cell_value(query.answers),
                    positions=QueryAnswer.get_boundaries(query.answers),
                    num_values=len(query.answers))
                )
            else:
                new_cells.append(old_cells[col_id])
        return TableRow(row_id, new_cells, row.text_id, row.text, row.relevant_text_boundaries)

    def get_join_dependent_queries(self, new_header, query, mask_entire_column=True):
        result_queries = list()
        for row_label, answers in QueryAnswer.group_by_surfaceform(query.answers).items():
            row_id = self.get_dependent_row_id(query, row_label)
            row_queries = list()
            found_query_with_answer = False
            answers_dict = QueryAnswer.get_dependent_queries(answers)
            for col_id in new_header.column_ids:
                if isinstance(col_id, tuple) and (mask_entire_column or col_id[1] in answers_dict):
                    new_query = self.get_dependent_query(
                        row_id, col_id,
                        answers=answers_dict.get(col_id[1], []),
                        dependency=None
                    )
                    row_queries.append(new_query)
                    found_query_with_answer = found_query_with_answer or col_id[1] in answers_dict
            if found_query_with_answer:
                result_queries.extend(row_queries)
        return result_queries

    def get_join_base_query(self, new_header, query):
        result_queries = list()

        row_id = self.get_dependent_row_id(query, None)
        query = deepcopy(query)
        query.row_id = row_id
        result_queries.append(query)

        for _, answers in QueryAnswer.group_by_surfaceform(query.answers).items():
            answers_dict = QueryAnswer.get_dependent_queries(answers)
            for col_id in new_header.column_ids:
                if isinstance(col_id, tuple):
                    new_query = self.get_dependent_query(
                        row_id, col_id,
                        answers=answers_dict.get(col_id[1], []),
                        dependency=QueryDependency(query, QueryAnswer.select_first_answer(answers))
                    )
                    result_queries.append(new_query)
        return result_queries

    def get_dependent_row_id(self, query, row_label):
        row_label = row_label or "?"
        row_id = f"{query.row_id}:{query.col_id}={row_label}"
        return row_id
    
    def group_dependent_query_rows(self, result_rows, result_queries):
        self.rng.shuffle(result_rows)
        for i in range(0, len(result_rows), 3):
            result_rows = result_rows[i: i + 3]
            row_ids = set(r.id for r in result_rows)
            result_queries = [q for q in result_queries if q.row_id in row_ids]
            if result_queries:
                yield result_rows, result_queries

    def get_dependent_query(self, row_id, col_id, answers, dependency):
        answer_data = {
            "answer_uri": [],
            "answer_normalized": [],
            "answer_start": [],
            "answer_end": [],
            "answer_surfaceform": [],
        }

        for answer in answers:
            for mention in answer["_answer_mentions"]:
                for k in answer_data:
                    answer_data[k].append(answer[f"_{k}"] if f"_{k}" in answer else mention[f"_{k}"])
        answers = QueryAnswer.many(**answer_data)
        return Query(row_id, col_id, answers, dependency=dependency)
