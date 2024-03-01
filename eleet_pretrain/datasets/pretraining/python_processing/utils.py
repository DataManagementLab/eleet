import logging
import numpy as np
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import SUBJ_INDEX
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import (ANSWER_END, ANSWER_NORMALIZED,
                                           ANSWER_START, ANSWER_SURFACEFORM, ANSWER_URI,
                                           ATTR_LABEL, ATTR_URI, DOC_NUMBER, DOC_URI, HINT_OBJ, HINT_PRED_LABEL,
                                           HINT_PRED_URI, SUBJECT_URI, SUBJ_INDEX, DOC_INDEX, 
                                           ATTR_DESCRIPTION)

logger = logging.getLogger(__name__)

TEST_SPLIT_DEFINITIONS = [
    ("nobel", "Q7191", "neighbors.o", {
        "let": {"prize": "$wikidata_id"},
        "pipeline": [
            {"$match": {"$expr": {"$in": [{"o": "$$prize", "p": "P166"}, "$neighbors"]}}}
        ],
    }),
    ("countries", "Q6256", "types", {}),
    ("skyscrapers", "Q11303", "types", {})
]
UNSEEN_QUERY_ATTRS = [
    "P1073",  # writable file format
    "P1046",  # discovery method
    "P2546",  # sidekick of
    "P194",  # legislative body
    "P1924",  # vaccine for
    "P567",  # underlies
    "P2155",  # solid solution series with
    "P744",  # asteroid family
    "P3461",  # designated as terrorist by
    "P3701",  # incarnation of
    "P263",  # official residence
    "P1322",  # dual to
    "P823",  # speaker'
    "P1321",  # place origin f o(Switzerland)
    "P1308",  # officeholder
    "P1313",  # office held by head of government
    "P1068",  # instruction set
    "P568",  # overlies
    "P520",  # armament
]


SPLIT_TEST_ALL = "split-all-test"

ATTR_COLS = [ATTR_LABEL, ATTR_URI, ATTR_DESCRIPTION]
DOC_COLS = [DOC_URI, DOC_NUMBER, DOC_INDEX]
SUBJECT_COLS = [SUBJECT_URI, SUBJ_INDEX]
HINT_COLS = [HINT_OBJ, HINT_PRED_URI, HINT_PRED_LABEL]
ANSWER_COLS = [ANSWER_START, ANSWER_END, ANSWER_URI, ANSWER_SURFACEFORM, ANSWER_NORMALIZED]
SPLITS = {0: "train", 1: "unseen_query", 2: "development",
          **{3 + si: s for si, (s, _, _, _) in enumerate(TEST_SPLIT_DEFINITIONS)}}
NUM_COLS = 10
NUM_OVERLAP_COLS = 5
LABEL_COLUMN = True
MAX_NUM_QUERY_ATTRS_JOIN = 3
MAX_NUM_QUERY_ATTRS_PRETRAIN = 5  # TODO move to Config
MAX_NUM_MASKED_PRETRAIN = 10

LABEL_NAMES = ["name", "label", "identifier", "alias", "aka", "also known as"]


def shorten_uri(uri):
    return uri.split("^^http")[0].split("/")[-1]


def rand_term(aliases, rng, entity_id=None, entity_uri=None, labels=None, label=None):
    assert labels is not None or label is not None
    assert entity_id is not None or entity_uri is not None
    if entity_id is None:
        entity_id = shorten_uri(entity_uri)
    if label is None:
        label = labels.get(entity_id, entity_id)
    assert isinstance(label, str)
    return rng.choice([label] + aliases.get(entity_id, []))


def to_dict(entries, key, unique=True, value=None, value_type=lambda x: x, access_method=dict.__getitem__):
    if entries is None:
        return dict()
    if unique:
        return {access_method(e, key): value_type(e if value is None else e[value]) for e in entries}
    else:
        result = dict()
        for e in entries:
            if access_method(e, key) not in result:
                result[access_method(e, key)] = []
            result[access_method(e, key)].append(e if value is None else access_method(e, value))
        return {k: value_type(v) for k, v in result.items()}


def group(rng, elements, max_group_size):
    shuffled = np.arange(len(elements))
    rng.shuffle(shuffled)
    num_subsets = int(np.ceil(len(elements) / max_group_size))
    subset_size = int(np.ceil(len(elements) / num_subsets))
    for i in range(0, len(elements), subset_size):
        subset = shuffled[i: i + subset_size]
        yield [elements[k] for k in subset]


class TableHeader():
    def __init__(self, column_names, column_ids):
        self._column_names = column_names
        self._column_ids = column_ids

    @property
    def column_names(self):
        return self._column_names

    @property
    def column_ids(self):
        return self._column_ids

    @property
    def column_ids_str(self):
        return [("-".join(c) if isinstance(c, tuple) else c) for c in self.column_ids]

    def to_dict(self):
        return dict(zip(self.column_ids, self.column_names))

    def __str__(self):
        return " | ".join(self.column_names)


class TableCell():
    def __init__(self, column_id, value, positions, num_values):
        self.column_id = column_id
        self.value = value
        self.positions = positions
        self.num_values = num_values

    @staticmethod
    def many(column_ids, values, positions, num_values):
        return list(map(lambda args: TableCell(*args), zip(column_ids, values, positions, num_values)))


class TableRow():
    def __init__(self, id, cells, text_id, text, relevant_text_boundaries):
        self.id = id
        self.cells = cells
        self.text_id = text_id
        self.text = text
        assert isinstance(relevant_text_boundaries, tuple) and len(relevant_text_boundaries) == 2 and \
            all(isinstance(e, int) for e in relevant_text_boundaries)
        self.relevant_text_boundaries = relevant_text_boundaries

    def __str__(self):
        return " | ".join(map(str, (c.value for c in self.cells))) + "\n\t" + self.text

    @property
    def values(self):
        return [c.value for c in self.cells]

    @property
    def num_values(self):
        return [c.num_values for c in self.cells]

    @property
    def positions(self):
        return [c.positions for c in self.cells]

    @staticmethod
    def get_mapping(rows):
        return {r.id: r for r in rows}

    @staticmethod
    def print_many(rows):
        for r in rows:
            print(r)


class Query():
    def __init__(self, row_id, col_id, answers, dependency=None):
        self.row_id = row_id
        self.col_id = col_id
        self.answers = answers
        self.dependency = dependency
        self.query_id = None

    @property
    def id(self):
        return self.row_id, self.col_id

    @property
    def col_id_str(self):
        return "-".join(self.col_id) if isinstance(self.col_id, tuple) else self.col_id

    @staticmethod
    def group_by_column(queries, unique=False):
        return to_dict(queries, key="col_id", unique=unique, access_method=getattr)

    @staticmethod
    def group_by_row(queries, unique=False):
        return to_dict(queries, key="row_id", unique=unique, access_method=getattr)

    @staticmethod
    def iter_with_query_id(queries):
        queries = list(queries)
        for i, q in enumerate(queries):
            q.query_id = i
        for q in queries:
            yield q.query_id, q


class QueryAnswer():
    def __init__(self, uri, normalized, surfaceform, start, end, dependent_query_data=None):
        self.uri = uri
        self.normalized = normalized
        self.start = start
        self.end = end
        self.surfaceform = surfaceform
        self.dependent_query_data = dependent_query_data

    @property
    def numeric_id(self):
        try:
            return int("1" + self.uri.split("/")[-1][1:])
        except ValueError:
            if self.uri.endswith("#dateTime"):
                return int("2" + "".join(c for c in self.uri.split("^^")[0] if c.isdigit()))
            else:
                raise NotImplementedError

    @staticmethod
    def many(answer_uri, answer_normalized, answer_surfaceform, answer_start, answer_end, dependent_queries=None):
        dependent_queries = dependent_queries or [None] * len(answer_uri)
        return list(map(lambda args: QueryAnswer(*args), zip(
            answer_uri, answer_normalized, answer_surfaceform, answer_start, answer_end, dependent_queries
        )))
    
    @staticmethod
    def group_by_surfaceform(answers):
        # mapping from surfaceform to all uris and token positions that have the surfaceform
        return to_dict(answers, "surfaceform", unique=False, access_method=getattr)

    @staticmethod
    def get_dependent_queries(answers):
        col_id_dependent_answer_pairs = [(d["_id"]["_attr_uri"].split("/")[-1], x)
                                         for a in answers for d in a.dependent_query_data for x in d["_answers"]]
        return to_dict(col_id_dependent_answer_pairs, key=0, unique=False, value=1, access_method=tuple.__getitem__)

    @staticmethod
    def select_first_answer(answers):
        first = answers[0]
        for a in answers[1:]:
            if a.start < first.start:
                first = a
        return first

    @staticmethod
    def get_boundaries(answers):
        return [(a.start, a.end) for a in answers]

    @staticmethod
    def get_cell_value(answers):
        return ", ".join(sorted(set(a.normalized for a in answers)))


class QueryDependency():
    def __init__(self, query, answer):
        self.query = query
        self.answer = answer
