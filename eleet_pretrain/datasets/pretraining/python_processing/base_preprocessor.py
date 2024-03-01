import numpy as np
import spacy
import logging
from copy import deepcopy
from eleet_pretrain.datasets.pretraining.data_import.wikidata_utils import WikidataProperties, shorten_uri
from eleet_pretrain.datasets.pretraining.python_processing.utils import rand_term, to_dict
from eleet_pretrain.datasets.pretraining.python_processing.utils import LABEL_COLUMN, LABEL_NAMES, NUM_COLS, NUM_OVERLAP_COLS, \
    rand_term, to_dict
from eleet_pretrain.datasets.pretraining.python_processing.utils import Query, TableHeader, TableRow, TableCell, QueryAnswer

AUGMENT_FACTOR = 3

logger = logging.getLogger(__name__)


class BasePreprocessor():
    def __init__(self, labels, aliases, relevant_properties, rng):
        super().__init__()
        # self.nlp = spacy.load('en_core_web_sm')  # TODO better but slower
        self.nlp = spacy.lang.en.English()
        self.nlp.add_pipe('sentencizer')
        self.relevant_properties = relevant_properties
        self.rng = rng
        self.labels = labels
        self.aliases = aliases

    def group_docs(self, docs_orig, max_group_size=3, min_group_size=2, do_augment=False, augment_factor=AUGMENT_FACTOR,
                   skip_too_small_groups=False):
        if not do_augment:
            augment_factor = 1
        docs = docs_orig["docs"]
        for _ in range(augment_factor):
            shuffled_idx = np.arange(len(docs))
            self.rng.shuffle(shuffled_idx)
            while True:
                if skip_too_small_groups and len(shuffled_idx) < min_group_size:
                    break  # happens if docs_orig is too small or dot dividable by given group sizes
                elif len(shuffled_idx) - max_group_size > min_group_size:  # emit max size group
                    yield deepcopy([docs[i] for i in shuffled_idx[:max_group_size]])
                    shuffled_idx = shuffled_idx[max_group_size:]
                elif len(shuffled_idx) <= max_group_size:  # emit what is left over
                    yield deepcopy([docs[i] for i in shuffled_idx])
                    break
                else:  # emit smallest possible group
                    yield deepcopy([docs[i] for i in shuffled_idx[:min_group_size]])
                    shuffled_idx = shuffled_idx[min_group_size:]

    def compute_queries(self, docs, rows, query_set):
        result = list()
        for doc, row in zip(docs, rows):
            assert doc["_id"]["_doc"] == row.id
            queries = {shorten_uri(k): v for k, v in  to_dict(doc["_queries"], key="_attr_uri").items()}
            for col_id in query_set:
                query_def = queries.get(col_id)
                if query_def is None:
                    continue
                query = self.to_query(query_def, row.id)
                result.append(query)
        return result

    def to_query(self, query_def, row_id):
        col_id = shorten_uri(query_def["_attr_uri"])
        answer_data = {
                    "answer_uri": [],
                    "answer_normalized": [],
                    "answer_start": [],
                    "answer_end": [],
                    "answer_surfaceform": [],
                    "dependent_queries": []
                }
                
        for answer in query_def["_answers"]:
            for mention in answer["_answer_mentions"]:
                for k in answer_data:
                    answer_data[k].append(answer[f"_{k}"] if f"_{k}" in answer else mention[f"_{k}"])
        return Query(row_id, col_id, QueryAnswer.many(**answer_data))

    def process_table(self, docs, query_set, num_overlap=NUM_OVERLAP_COLS, num_cols=NUM_COLS,
                      fixed_columns=tuple(), random_names=True):
        evidences, result_docs = list(), list()
        columns = list(fixed_columns)  # TODO stats
        if not fixed_columns:
            column_options = self.get_column_options(docs)
            columns = self.choose_columns(docs, column_options,
                                          num_overlap=num_overlap,
                                          num_cols=num_cols, include=query_set)
        column_names = self.get_column_names(columns, num_cols=num_cols, random=random_names)
        column_ids = list(columns)
        if LABEL_COLUMN:
            column_names = [self.rng.choice(LABEL_NAMES) if random_names else LABEL_NAMES[0]] + column_names
            column_ids = ["id"] + column_ids
        for doc in docs:
            for evidence in doc["evidence"]:
                if "properties" not in evidence:
                    logger.warn(f"Skipping evidence {evidence}")
                    continue
                values, num_values = self.get_column_values(columns, evidence, num_cols=num_cols)  # TODO --> add to collected data
                value_positions = self.get_covered_positions(doc, columns, evidence)


                if LABEL_COLUMN:
                    values = [rand_term(aliases=self.aliases, label=evidence["label"],
                                        entity_id=evidence["wikidata_id"], rng=self.rng)
                              if random_names else evidence["label"]] + values
                    value_positions = self.get_covered_positions(doc, ["id"], evidence) + value_positions
                    num_values = [1] + num_values
                
                if any(values):
                    cells = TableCell.many(column_ids, values, value_positions, num_values)
                    evidences.append(TableRow(
                        id=doc["_id"]["_doc"], cells=cells, text_id=doc["text"]["text_idx"], text=doc["text"]["text"],
                        relevant_text_boundaries=(doc["text"]["relevant_text_start"], doc["text"]["relevant_text_end"]),
                    ))
                    result_docs.append(doc)
        return result_docs, evidences, TableHeader(column_names, column_ids)

    def choose_columns(self, docs, column_options, num_cols, num_overlap, include=[]):
        include = set(include) & (column_options.keys())
        column_options = {k: v for k, v in column_options.items() if k not in include}

        covers_dict = to_dict([x for doc in docs for x in doc["text"]["covers"]], "s", False, "p")
        covered_properties = set(x for doc in docs for e in doc["evidence"]
                                 for x in covers_dict.get(e["wikidata_id"], []))
        covered_options = covered_properties & set(column_options.keys())
        
        non_covered_options = set(column_options.keys()) - covered_options
        num_covered = max(0, min(len(covered_options), num_overlap, num_cols - len(include)))
        num_non_covered = max(0, min(len(non_covered_options), num_cols - num_covered - len(include)))
        columns = sorted(include)[:num_cols]
        columns += list(self.rng.choice(np.array(sorted(covered_options, key=str), dtype=object),
                                        num_covered, replace=False))
        if non_covered_options:
            p = np.array([column_options[x] for x in sorted(non_covered_options, key=str)])
            p = p / np.sum(p)
            columns += list(self.rng.choice(np.array(sorted(non_covered_options, key=str), dtype=object),
                                       num_non_covered, p=p, replace=False))
        columns = [(tuple(x) if isinstance(x, np.ndarray) else x) for x in columns]
        return columns

    def get_covered_positions(self, doc, columns, evidence):
        positions_dict = {p: [(x["start"], x["end"]) for x in v]
            for p, v in to_dict(to_dict(doc["text"]["covers"], "s", False).get(
                evidence["wikidata_id"]), "p", False).items()}
        return [positions_dict.get(c, []) for c in columns]

    def get_column_names(self, columns, num_cols, random):
        column_names = list()
        for c in columns:
            if random:
                column_names.append(rand_term(labels=self.labels, aliases=self.aliases, entity_id=c, rng=self.rng))
            else:
                column_names.append(self.labels.get(c, c))
        column_names += [None] * max(0, (num_cols - len(column_names)))
        return column_names

    def get_column_values(self, columns, evidence, num_cols):
        properties = WikidataProperties(evidence["properties"], self.labels, self.aliases, self.rng)
        values = []
        num_values = []
        for c in columns:
            if c in properties:
                v = list(properties[c].iter(filter_none=True, convert="rand_term"))
                values.append(", ".join(v))
                num_values.append(len(v))
            else:
                values.append(None)
                num_values.append(0)
        values += [None] * (num_cols - len(values))
        num_values += [0] * (num_cols - len(values))
        return values, num_values

    def get_column_options(self, docs, exclude=[]):
        options = dict()  # TODO handle include
        for doc in docs:
            for evidence in doc["evidence"]:
                if "properties" not in evidence:
                    continue
                properties = WikidataProperties(evidence["properties"], self.labels, self.aliases, self.rng)
                exclude_properties = set(shorten_uri(q["_attr_uri"]) for q in exclude)
                for p in sorted((properties.keys() & self.relevant_properties) - exclude_properties):
                    options[p] = options.get(p, 0)
                    options[p] += 1
        options = {k: 2 ** v for k, v in options.items()}
        return options
