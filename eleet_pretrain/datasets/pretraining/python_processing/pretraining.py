"""Classes to collect data."""

import logging
from copy import deepcopy
from math import ceil

import numpy as np
from eleet_pretrain.datasets.pretraining.data_import.wikidata_utils import shorten_uri
from eleet_pretrain.datasets.pretraining.python_processing.base_preprocessor import BasePreprocessor
from eleet_pretrain.datasets.pretraining.python_processing.dependent_queries_mixin import DependentQueriesMixin, DQ_MODE
from eleet_pretrain.datasets.pretraining.python_processing.utils import MAX_NUM_MASKED_PRETRAIN, \
    MAX_NUM_QUERY_ATTRS_PRETRAIN, UNSEEN_QUERY_ATTRS, Query, QueryAnswer, group, to_dict
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI

logger = logging.getLogger(__name__)

MASK_ROW_LABEL_PROB = 0.15
EMPTY_QUERY_PROB = 1/3


class PretrainingDataPreprocessor(BasePreprocessor, DependentQueriesMixin):

    def construct_dataset(self, docs, result_data, evidences_dict, texts_dict, table_name, split, do_augment):
         for j, doc_group in enumerate(self.group_docs(docs, do_augment=do_augment)):
            self.process(docs=doc_group, result_data=result_data, evidences_dict=evidences_dict,
                         texts_dict=texts_dict, table_name=f"{table_name}-{j}", split=split)

    def process(self, docs, result_data, evidences_dict, texts_dict, table_name, split):
        self.apply_text_and_evidence(docs, texts_dict, evidences_dict)
        for i, (query_set, remaining) in enumerate(self.group_queries(docs)):
            docs, rows, header = self.process_table(docs, query_set=query_set)
            query_set = [q for q in query_set if q in header.column_ids]
            remaining_set = sorted(set(header.column_ids) & remaining)
            if not query_set:
                continue
            queries = self.compute_queries(docs, rows, query_set=query_set)
            row_label_queries = list(self.get_row_label_queries(docs, rows))
            align_text_queries_basic = self.compute_queries(docs, rows, query_set=remaining_set)  # will not be masked

            for suffix, kwargs in self.process_regular_queries(
                docs, header, rows, queries, align_text_queries_basic, row_label_queries
            ):
                result_data.append(table_name=f"{table_name}-{i}-{suffix}", **kwargs)
            
            for suffix, kwargs in self.process_dependent_queries(
                docs, header, rows, queries, align_text_queries_basic, row_label_queries
            ):
                result_data.append(table_name=f"{table_name}-{i}-{suffix}", **kwargs)

    def process_regular_queries(self, docs, header, rows, queries, align_text_queries_basic, row_label_queries):
        for j, query_subset in enumerate(group(self.rng, queries, MAX_NUM_MASKED_PRETRAIN)):
            remaining_subset = set([q.id for q in queries]) - set([q.id for q in query_subset])
            align_text_queries = align_text_queries_basic + list(q for q in queries if q.id in remaining_subset)
            empty_queries = self.generate_empty_queries(rows)
            header_queries = self.get_header_queries(
                rows, docs, query_subset + align_text_queries + row_label_queries + empty_queries,
                is_dependent_queries=False
            )
            query_subset.extend(q for q in row_label_queries if self.rng.random() < MASK_ROW_LABEL_PROB)
            yield j, {
                "rows": rows,
                "header": header,
                "queries": query_subset + empty_queries,
                "db_operator": "pretraining",
                "header_queries": header_queries
            }

    def process_dependent_queries(self, docs, header, rows, queries, align_text_queries_basic, row_label_queries):
        dependent_queries = self.compute_dependent_queries(header, rows, queries, DQ_MODE.PRETRAIN)
        for j, d in enumerate(dependent_queries):
            for a_suffix, a_header, a_rows, a_queries in self.dependent_to_pretrain_queries(*d):
                empty_queries, empty_queries_dep = self.generate_empty_queries(a_rows, return_dependent=True)
                header_queries = self.get_header_queries(a_rows, docs, d[-1] + empty_queries_dep,
                                                         is_dependent_queries=True)
                left_queries = [q for q in queries + align_text_queries_basic + row_label_queries
                                if q.col_id in a_header.column_ids]
                header_queries += self.get_header_queries(a_rows, docs, left_queries + empty_queries,
                                                          is_dependent_queries=False)
                a_queries.extend(Query(row_id=r.id, col_id=q.col_id, answers=q.answers)
                                 for q in row_label_queries
                                 if self.rng.random() < MASK_ROW_LABEL_PROB and q.col_id in a_header.column_ids
                                 for r in a_rows if r.id.split(":")[0] == q.row_id)
                yield f"{j}-{a_suffix}", {
                    "rows": a_rows,
                    "header": a_header,
                    "queries": a_queries + empty_queries + empty_queries_dep,
                    "db_operator": "pretraining_dependent_queries",
                    "header_queries": header_queries
                }

    def generate_empty_queries(self, rows, return_dependent=False):
        result_regular = list()
        result_dependent = list()
        for row in rows:
            empty_cell_ids = np.where(np.array(row.num_values) == 0)[0]
            decision = self.rng.random(*empty_cell_ids.shape) < EMPTY_QUERY_PROB
            for cell_id in empty_cell_ids[decision]:
                col_id = row.cells[cell_id].column_id
                if col_id in UNSEEN_QUERY_ATTRS:
                    continue
                query = Query(row_id=row.id, col_id=col_id, answers=[])
                (result_dependent if isinstance(col_id, tuple) else result_regular).append(query)
        if return_dependent:
            return result_regular, result_dependent
        assert len(result_dependent) == 0
        return result_regular

    def apply_text_and_evidence(self, docs, texts_dict, evidences_dict):
        shuffled = np.arange(len(docs))
        self.rng.shuffle(shuffled)
        docs = [docs[i] for i in shuffled]
        for doc in docs:
            doc["evidence"] = [deepcopy(evidences_dict[doc["_evidence"]])]
            doc["text"] = deepcopy(texts_dict[doc["text"]["text_idx"]])
            doc["text"]["relevant_text_start"] = 0
            doc["text"]["relevant_text_end"] = len(doc["text"]["text"])
            assert not doc["text"]["is_test"]
            self.add_confusion_text(doc)

    def add_confusion_text(self, doc):
        if "confusion" not in doc["text"]:
            return

        r = self.rng.random()
        if r < 1 / 3:  # append confusion
            offset = len(doc["text"]["text"]) + 1
            doc["text"]["text"] = doc["text"]["text"] + " " + doc["text"]["confusion"]
            doc["text"]["text_idx"] = doc["text"]["text_idx"] + "-" + doc["text"]["confusion_idx"]

            for c in doc["text"]["confusion_covers"]:
                c["start"] += offset
                c["end"] += offset
        elif r < 2 / 3:  # prepend confusion
            offset = len(doc["text"]["confusion"]) + 1
            doc["text"]["text"] = doc["text"]["confusion"] + " " + doc["text"]["text"]
            doc["text"]["text_idx"] = doc["text"]["confusion_idx"] + "-" + doc["text"]["text_idx"]

            doc["text"]["relevant_text_start"] += offset
            doc["text"]["relevant_text_end"] += offset

            for c in doc["text"]["covers"]:
                c["start"] += offset
                c["end"] += offset

            for m in (m for q in doc["_queries"] for a in q["_answers"] for m in a["_answer_mentions"]):
                m["_answer_start"] += offset
                m["_answer_end"] += offset

            for dm in (dm for q in doc["_queries"] for a in q["_answers"] for d in a["_dependent_queries"]
                       for da in d["_answers"] for dm in da["_answer_mentions"]):
                dm["_answer_start"] += offset
                dm["_answer_end"] += offset

    def group_queries(self, docs):
        query_attrs = sorted(set([shorten_uri(q[ATTR_URI]) for doc in docs for q in doc["_queries"]]) - {"id"})
        for g in  group(self.rng, query_attrs, MAX_NUM_QUERY_ATTRS_PRETRAIN):
            yield g, set(query_attrs) - set(g)
    
    def get_row_label_queries(self, docs, rows):
        for doc, row in zip(docs, rows):
            row_label_query = [q for q in doc["_queries"] if q[ATTR_URI] == "id"]
            if not row_label_query:
                continue
            yield self.to_query(row_label_query[0], row.id)

    def dependent_to_pretrain_queries(self, a_suffix, a_header, a_rows, a_queries):
        group_size = int(ceil(len(a_queries) / 2))
        for i, q in enumerate(group(self.rng, a_queries, group_size)):
            yield f"{a_suffix}-{i}",  a_header, a_rows, q

    def get_header_queries(self, rows, docs, queries, is_dependent_queries):
        """For the given queries, generate header queries: Incorporate results from confusion text."""
        schema_queries = []
        queried_cols = sorted(set(q.col_id for q in queries))

        for c in queried_cols:
            for doc in docs:

                if "confusion_idx" in doc["text"] and "-" in doc["text"]["text_idx"]:  # confusion text appended
                    covers = to_dict(doc["text"]["covers"] + doc["text"]["confusion_covers"], key="p", unique=False)
                    subjects = set([doc["text"]["confusion_idx"], doc["_id"]["_doc"]])
                else:
                    covers = to_dict(doc["text"]["covers"], key="p", unique=False)
                    subjects = set([doc["_id"]["_doc"]])

                query_predicate = c
                if is_dependent_queries:
                    subjects = set([shorten_uri(x["o"]) for x in covers.get(c[0], []) if x["s"] in subjects])
                    query_predicate = c[1]

                answers = [q for q in covers.get(query_predicate, []) if q["s"] in subjects]
                answers = [
                    QueryAnswer(o, None, None, start, end)
                    for o, start, end in set((a["o"], a["start"], a["end"]) for a in answers)
                ]
                if len(answers) == 0:
                    continue
                for r in rows:
                    if r.id.split(":")[0] == doc["_id"]["_doc"]:
                        qq = Query(row_id=r.id, col_id=c, answers=answers)
                        schema_queries.append(qq)
        return schema_queries
