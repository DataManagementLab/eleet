import pandas as pd
from eleet_pretrain.datasets.pretraining.python_processing.utils import Query

OPERATOR_DICT = {
    "pretraining": 0,
    "default_join": 1,
    "multi_join": 2,
    "default_union": 3,
    "multi_union": 4,
    "default_join_dependent_queries": 5,
    "multi_join_dependent_queries": 6,
    "pretraining_dependent_queries": 7
}

class CollectedData():
    def __init__(self, aliases, rng, prefix):
        self.aliases = aliases
        self.rng = rng
        self._datasets = {}
        self.prefix = prefix

    def reset(self):
        for d in self._datasets.values():
            d.reset()

    def is_empty(self):
        return all(d.is_empty() for d in self._datasets.values())

    @property
    def datasets(self):
        return self._datasets.keys()
    
    def items(self, dataset_name="default"):
        return self._datasets[dataset_name].items()

    def append(self, table_name, rows, header, queries, db_operator, dataset_name="default", header_queries=[]):
        if dataset_name not in self._datasets:
            self._datasets[dataset_name] = CollectedDataset(self.aliases, self.rng,
                                                            prefix=f"{self.prefix}-{len(self._datasets)}")
        return self._datasets[dataset_name].append(table_name, rows, header, queries, header_queries, db_operator)


class CollectedDataset():
    def __init__(self, aliases, rng, prefix):
        self._current_table_id = 0
        self.rows = list()
        self.row_meta = list()
        self.overlap_mentions = list()
        self.texts = list()
        self.row_idx = list()
        self.overlap_mentions_idx = list()

        self.header_columns = list()
        self.header_column_ids = list()
        self.header_meta = list()
        self.header_idx = list()

        self.queries = list()
        self.queries_idx = list()
        self.answers = list()
        self.answer_idx = list()
        # self.align_text = list()
        # self.align_text_idx = list()
        self.header_queries = list()
        self.header_queries_idx = list()

        self.aliases = aliases
        self.rng = rng
        self.prefix = prefix

    @property
    def current_table_id(self):
        return f"{self.prefix}-{self._current_table_id}"

    def reset(self):
        old_table_id = self._current_table_id
        self.__init__(self.aliases, self.rng, self.prefix)
        self._current_table_id = old_table_id

    def is_empty(self):
        return len(self.rows) == 0

    def items(self):
        self.create_dataframes()
        yield "rows", self.rows
        yield "row_meta", self.row_meta
        yield "overlap_mentions", self.overlap_mentions
        yield "texts", self.texts

        yield "header_columns", self.header_columns
        yield "header_column_ids", self.header_column_ids
        yield "header_meta", self.header_meta

        yield "queries", self.queries
        yield "answers", self.answers
        # yield "align_text", self.align_text
        yield "header_queries", self.header_queries
        self.reset()

    def append(self, table_name, rows, header, queries, header_queries, db_operator):
        self.header_columns.append(header.column_names)
        self.header_column_ids.append(header.column_ids_str)
        self.header_meta.append((table_name, OPERATOR_DICT[db_operator]))
        self.header_idx.append(self.current_table_id)

        for row in rows:
            self.rows.append(row.values)
            self.texts.append([row.text_id, row.text, *row.relevant_text_boundaries])
            self.row_meta.append((row.num_values, ))
            self.row_idx.append((self.current_table_id, row.id))
            for positions, col_id in zip(row.positions, header.column_ids_str):
                for i, pos in enumerate(positions):
                    self.overlap_mentions.append(pos)
                    self.overlap_mentions_idx.append((self.current_table_id, row.id, col_id, i))

        assert any([q.answers for q in queries])
        for q_id, query in Query.iter_with_query_id(queries):
            self.queries_idx.append((self.current_table_id, query.row_id, query.col_id_str, q_id))

            if query.dependency:
                self.queries.append((query.dependency.query.query_id, query.dependency.answer.start))
            else:
                self.queries.append((-1, -1))

            for i, answer in enumerate(query.answers):
                self.answers.append((answer.start, answer.end, answer.numeric_id, answer.normalized, answer.surfaceform))
                self.answer_idx.append((self.current_table_id, query.row_id, query.col_id_str, q_id, i))

        for query in header_queries:
            for i, answer in enumerate(query.answers):
                self.header_queries.append((answer.start, answer.end, answer.numeric_id))
                self.header_queries_idx.append((self.current_table_id, query.row_id, query.col_id_str, i))
        self._current_table_id += 1

    def create_dataframes(self):
        self.rows = pd.DataFrame(
            self.rows,
            index=pd.MultiIndex.from_tuples(self.row_idx, names=("table_id", "row_id"))
        )
        self.row_meta = pd.DataFrame(
            self.row_meta,
            columns=("num_values",),
            index=pd.MultiIndex.from_tuples(self.row_idx, names=("table_id", "row_id"))
        )
        self.texts = pd.DataFrame(
            self.texts,
            columns=("text_id", "text", "relevant_text_start", "relevant_text_end"),
            index=pd.MultiIndex.from_tuples(self.row_idx, names=("table_id", "row_id"))
        )
        self.rows.columns = [str(c) for c in self.rows.columns]
        self.overlap_mentions = pd.DataFrame(
            self.overlap_mentions,
            index=pd.MultiIndex.from_tuples(self.overlap_mentions_idx, names=("table_id", "row_id", "col_id", "mention_id")),
            columns=["mention_start", "mention_end"]
        )

        self.header_columns = pd.DataFrame(self.header_columns, index=self.header_idx)
        self.header_columns.columns = [str(c) for c in self.header_columns.columns]
        self.header_column_ids = pd.DataFrame(self.header_column_ids, index=self.header_idx)
        self.header_column_ids.columns = [str(c) for c in self.header_column_ids.columns]
        self.header_meta = pd.DataFrame(self.header_meta, index=self.header_idx, columns=("table_name", "db_operator"))

        self.queries = pd.DataFrame(
            self.queries,
            index=pd.MultiIndex.from_tuples(self.queries_idx, names=("table_id", "row_id", "col_id", "query_id")),
            columns=("dependency_query", "dependency_answer_start")
        )
        self.answers = pd.DataFrame(
            self.answers,
            index=pd.MultiIndex.from_tuples(self.answer_idx, names=("table_id", "row_id", "col_id", "query_id",
                                                                    "answer_id")),
            columns=("answer_start", "answer_end", "answer_numeric_id", "answer_normalized", "answer_surfaceform")
        )
        self.header_queries = pd.DataFrame(
            self.header_queries,
            index=pd.MultiIndex.from_tuples(self.header_queries_idx, names=("table_id", "row_id", "col_id", "query_id")),
            columns=("answer_start", "answer_end", "answer_numeric_id")
        )
        # self.align_text = pd.DataFrame(
        #     self.align_text,
        #     index=pd.MultiIndex.from_tuples(self.align_text_idx, names=("table_id", "row_id", "col_id", "align_id")),
        #     columns=("align_start", "align_end")
        # )
