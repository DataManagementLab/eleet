"""Classes to collect data."""

import logging
from copy import deepcopy
import re

import numpy as np
from eleet_pretrain.datasets.pretraining.python_processing.base_preprocessor import BasePreprocessor
from eleet_pretrain.datasets.pretraining.python_processing.dependent_queries_mixin import DependentQueriesMixin, DQ_MODE
from eleet_pretrain.datasets.pretraining.python_processing.utils import MAX_NUM_QUERY_ATTRS_JOIN, Query, QueryAnswer
from eleet_pretrain.datasets.pretraining.data_import.wikidata_utils import shorten_uri
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ATTR_URI

logger = logging.getLogger(__name__)
FIXED_TABLES = {
    "nobel": {
        "person": ("P19", "P569", "P20", "P570", "P27", "P103", "P21", "P551", "P1412", "P735"),
        "reward": ("P101", "P106", "P108", "P166", "P1411", "P463", "P69", "P184", "P185", "P1344")
    },
    "countries": {
        "geographical": ("P30", "P36", "P47", "P361", "P1589", "P610", "P206", "P706", "P31", "P37"),
        "political": ("P35", "P6", "P122", "P571", "P463", "P530", "P150", "P138", "P1365", "P155")
    },
    "skyscrapers": {
        "location+people": ("P17", "P625", "P131", "P84", "P193", "P669", "P276", "P127", "P631", "P88"),
        "dates+style": ("P571", "P1619", "P149", "P366", "P1435", "P466", "P793", "P361", "P186", "P138")
    }
}  # TODO fix unseen queries as well

class DB_OPERATION_TYPE():
    DEFAULT_JOIN = "default_join"
    MULTI_JOIN = "multi_join"
    DEFAULT_UNION = "default_union"
    MULTI_UNION = "multi_union"

JOIN_OPERATIONS = (DB_OPERATION_TYPE.DEFAULT_JOIN, DB_OPERATION_TYPE.MULTI_JOIN)
UNION_OPERATIONS = (DB_OPERATION_TYPE.DEFAULT_UNION, DB_OPERATION_TYPE.MULTI_UNION)
DB_OPERATIONS = JOIN_OPERATIONS + UNION_OPERATIONS
GROUP_PARAMS = {DB_OPERATION_TYPE.MULTI_UNION: {
    "max_group_size": 4,
    "min_group_size": 3,
    "skip_too_small_groups": True
}}
FINETUNE_TEST_SPLIT_SIZE = 0.3

class TestingDataPreprocessor(BasePreprocessor, DependentQueriesMixin):
    def construct_dataset(self, docs, result_data, evidences_dict, texts_dict, table_name, split, do_augment):
        for j, (db_operation, doc_group) in enumerate(self.generate_db_operations(docs, texts_dict,
                                                                                  evidences_dict, split)):
            self.process(db_operation=db_operation, docs=doc_group, result_data=result_data,
                         table_name=f"{table_name}-{db_operation}-{j}", split=split)

    def generate_db_operations(self, orig_docs, texts_dict, evidences_dict, split):
        apply_funcs = {
            DB_OPERATION_TYPE.DEFAULT_JOIN: self.apply_single,
            DB_OPERATION_TYPE.MULTI_JOIN: self.apply_multi,
            DB_OPERATION_TYPE.DEFAULT_UNION: self.apply_single,
            DB_OPERATION_TYPE.MULTI_UNION: self.apply_multi,
        }
        for op in DB_OPERATIONS:
            if split == "unseen_query" and op in UNION_OPERATIONS:
                continue

            for docs in self.group_docs(orig_docs, **GROUP_PARAMS.get(op, {})):
                docs = apply_funcs[op](docs, texts_dict, evidences_dict)
                yield op, docs

    def process(self, db_operation, docs, result_data, table_name, split):
        for j, (doc_idx, query_set, fixed_columns) in enumerate(self.group_queries(db_operation, docs, split)):
            docs, rows, header = self.process_table(docs, query_set=query_set, fixed_columns=fixed_columns,
                                                    random_names=False)
            query_set = [q for q in query_set if q in header.column_ids]
            if not query_set or query_set == ["id"]:
                continue
            queries = self.compute_queries(doc_idx, db_operation, docs, rows, header, query_set=query_set)
            if not queries:
                continue

            this_table_name = f"{table_name}-{db_operation}-{j}"

            if queries and db_operation != DB_OPERATION_TYPE.MULTI_UNION:
                result_data.append(table_name=this_table_name, rows=rows, header=header, queries=queries,
                                   db_operator=db_operation, dataset_name=db_operation)
            else:
                self.add_multi_union_queries(result_data, this_table_name, db_operation, rows, header, queries)

            if db_operation in JOIN_OPERATIONS:
                self.add_dependent_join_queries(this_table_name, db_operation, result_data, rows, header, queries)

    def add_multi_union_queries(self, result_data, table_name, db_operation, rows, header, queries):
        self._add_dependent(table_name=table_name,
                            db_operation=db_operation,
                            result_data=result_data,
                            rows=rows, header=header, queries=queries,
                            modes=(DQ_MODE.EVAL_TRAIN_MULTI_UNION, DQ_MODE.EVAL_TABLE_DECODING_MULTI_UNION),
                            dataset_suffixes=("_train", "_eval"),
                            operation_suffix="")

    def add_dependent_join_queries(self, table_name, db_operation, result_data, rows, header, queries):
        self._add_dependent(table_name=table_name,
                            db_operation=db_operation,
                            result_data=result_data,
                            rows=rows, header=header, queries=queries,
                            modes=(DQ_MODE.EVAL_TRAIN_JOIN, DQ_MODE.EVAL_TABLE_DECODING_JOIN),
                            dataset_suffixes=("_td_train", "_td_eval"),
                            operation_suffix="_dependent_queries")

    def _add_dependent(self, table_name, db_operation, result_data, rows, header, queries,
                       modes, dataset_suffixes, operation_suffix):
        x = 0 if self.rng.random() > FINETUNE_TEST_SPLIT_SIZE else 1
        mode = modes[x]
        dataset_name_suffix = dataset_suffixes[x]
        dq = self.compute_dependent_queries(header, rows, queries, mode=mode)

        for a_suffix, a_header, a_rows, a_queries in dq:
            result_data.append(f"{table_name}-{a_suffix}",
                               a_rows, a_header, a_queries, db_operation + operation_suffix,
                               dataset_name=db_operation + dataset_name_suffix)

    def compute_queries(self, doc_idx, op, docs, rows, header, query_set):
        if op in JOIN_OPERATIONS:
            queries =  super().compute_queries(docs, rows, query_set)
            queries = self.mask_entire_column(rows, queries, query_set)
        elif op == "multi_union":
            doc_pairs = self.get_multi_union_pairs(docs)
            row_pairs = self.get_multi_union_pairs(rows)
            queries = super().compute_queries(doc_pairs[doc_idx], row_pairs[doc_idx], query_set)
            for row in row_pairs[doc_idx]:
                queries = self.mask_entire_row(row, header, queries)
        else:
            queries = super().compute_queries([docs[doc_idx]], [rows[doc_idx]], query_set)
            queries = self.mask_entire_row(rows[doc_idx], header, queries)
        return queries

    def mask_entire_column(self, rows, queries, query_set):
        done = {(q.col_id, q.row_id) for q in queries}
        for row in rows:
            for q in query_set:
                if (q, row.id) in done:
                    continue
                queries.append(Query(
                    row_id=row.id,
                    col_id=q,
                    answers=[]
                ))
        return queries

    def mask_entire_row(self, row, header, queries):
        done = {q.col_id for q in queries if q.row_id == row.id}
        missing = set(header.column_ids) - done

        for m in missing:
            answers = []

            queries.append(Query(
                row_id=row.id,
                col_id=m,
                answers=answers
            ))  # TODO case where identifier mentions not given by TREx
        return queries

    def group_queries(self, op, docs, split):
        group_funcs = {
            DB_OPERATION_TYPE.DEFAULT_JOIN: self.group_join,
            DB_OPERATION_TYPE.MULTI_JOIN: self.group_join,
            DB_OPERATION_TYPE.DEFAULT_UNION: self.group_union,
            DB_OPERATION_TYPE.MULTI_UNION: self.group_multi_union,
        }
        return group_funcs[op](docs, split)

    def group_join(self, docs, split):
        for _, fixed_columns in FIXED_TABLES.get(split, {"default": list()}).items():
            fixed_columns = fixed_columns[:-MAX_NUM_QUERY_ATTRS_JOIN]

            query_attrs = set(shorten_uri(q[ATTR_URI]) for doc in docs for q in doc["_queries"])
            query_attrs = sorted(query_attrs - {"id"} - set(fixed_columns))
            for i in range(0, len(query_attrs), MAX_NUM_QUERY_ATTRS_JOIN):
                group = query_attrs[i: i + MAX_NUM_QUERY_ATTRS_JOIN]
                columns = (group + list(fixed_columns)) if fixed_columns else []
                yield None, group, columns

    def group_union(self, docs, split):
        for _, fixed_columns in FIXED_TABLES.get(split, {"default": list()}).items():
            for i, doc in enumerate(docs):
                available_attrs = set(shorten_uri(q[ATTR_URI]) for q in doc["_queries"] if q[ATTR_URI])
                query_attrs = fixed_columns or sorted(available_attrs)
                query_attrs = ["id"] + [a for a in query_attrs if a in available_attrs]
                yield i, query_attrs, fixed_columns

    def group_multi_union(self, docs, split):
        doc_pairs = self.get_multi_union_pairs(docs)
        for _, fixed_columns in FIXED_TABLES.get(split, {"default": list()}).items():
            for i, doc_pair in enumerate(doc_pairs):
                assert all(doc_pair[0]["text"]["text_idx"] == d["text"]["text_idx"] for d in doc_pair)
                available_attrs = set(shorten_uri(q[ATTR_URI])
                                      for doc in doc_pair
                                      for q in doc["_queries"] if q[ATTR_URI]
                                      if q[ATTR_URI] != "id")
                query_attrs = fixed_columns or sorted(available_attrs)
                query_attrs = ["id"] + [a for a in query_attrs if a in available_attrs]
                yield i, query_attrs, fixed_columns

    def get_multi_union_pairs(self, docs):
        doc_pairs = [tuple(docs[j] for j in range(i, i + 2) if j < len(docs)) for i in range(0, len(docs), 2)]
        return doc_pairs

    def apply_single(self, docs, texts_dict, evidences_dict):
        for doc in docs:
            doc["evidence"] = [evidences_dict[doc["_evidence"]]]
            doc["text"] = texts_dict[doc["text"]["text_idx"]]
            doc["text"]["relevant_text_start"], doc["text"]["relevant_text_end"] = 0, len(doc["text"]["text"])
        return docs

    def apply_multi(self, docs, texts_dict, evidences_dict):
        shuffled = np.arange(len(docs))
        self.rng.shuffle(shuffled)
        docs = [docs[i] for i in shuffled]
        for i in range(0, len(docs), 2):
            doc1, doc2 = docs[i], (docs[i + 1] if (i + 1) < len(docs) else None)
            doc1["evidence"] = [deepcopy(evidences_dict[doc1["_evidence"]])]
            doc1["text"] = deepcopy(texts_dict[doc1["text"]["text_idx"]])
            doc1["text"]["relevant_text_start"], doc1["text"]["relevant_text_end"] = 0, len(doc1["text"]["text"])

            if doc2:
                doc2["evidence"] = [evidences_dict[doc2["_evidence"]]]
                doc2["text"] = texts_dict[doc2["text"]["text_idx"]]
                self._merge_docs(doc1, doc2)
        return docs

    def _merge_docs(self, doc1, doc2):
        offset = len(doc1["text"]["text"]) + 1
        multi_text = doc1["text"]["text"] + " " + doc2["text"]["text"]
        multi_covers = doc1["text"]["covers"] + [
            {"s": x["s"], "p": x["p"],"start": x["start"] + offset, "end": x["end"] + offset}
            for x in doc2["text"]["covers"]
        ]
        multi_text_idx = doc1["text"]["text_idx"] + "-" + doc2["text"]["text_idx"]
        doc1["text"] = {"text": multi_text, "covers": multi_covers, "text_idx": multi_text_idx }
        doc2["text"] = deepcopy(doc1["text"])
        doc1["text"]["relevant_text_start"], doc1["text"]["relevant_text_end"] = 0, offset - 1 
        doc2["text"]["relevant_text_start"], doc2["text"]["relevant_text_end"] = offset, len(multi_text) 

        for m in (m for q in doc2["_queries"] for a in q["_answers"] for m in a["_answer_mentions"]):
            m["_answer_start"] += offset
            m["_answer_end"] += offset
        
        for dm in (dm for q in doc2["_queries"] for a in q["_answers"] for d in a["_dependent_queries"]
                   for da in d["_answers"] for dm in da["_answer_mentions"]):
            dm["_answer_start"] += offset
            dm["_answer_end"] += offset

    # def process_helper_examples(self, doc):
    #     if "helper_examples" not in doc:
    #         return ("", "", [], [], [])
    #     helper_examples = doc["helper_examples"]
    #     if helper_examples is None or len(helper_examples) == 0:
    #         return ("", "", [], [], [])
    #     attr_uri = helper_examples["_id"]
    #     result = ""
    #     start, end, norm = [], [], []
    #     for example in helper_examples["example"]:
    #         text = example["text"]
    #         sentences = list(self.nlp(text).sents)
    #         answers = defaultdict(list)
    #         answers_seen = set()
    #         for answer in example["query"]["_answers"]:
    #             if answer[ANSWER_NORMALIZED] in answers_seen:
    #                 continue
    #             answers_seen.add(answer[ANSWER_NORMALIZED])
    #             sentence = max(sentences, key=lambda x: (answer[ANSWER_START] >= x.start_char, x.start_char))
    #             answers[sentence.start_char].append(
    #                 (answer[ANSWER_START] - sentence.start_char,
    #                  answer[ANSWER_END] - sentence.start_char,
    #                  answer[ANSWER_NORMALIZED])
    #             )

    #         for sentence in sentences:
    #             a = answers[sentence.start_char]
    #             if len(a) == 0:
    #                 continue
    #             s, e, n = zip(*a)
    #             start += [x + len(result) for x in s]
    #             end += [x + len(result) for x in e]
    #             norm += n
    #             result += str(sentence) + " "
    #     return (attr_uri, result, start, end, norm)