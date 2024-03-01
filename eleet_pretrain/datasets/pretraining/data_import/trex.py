"""Load Trex-dataset into tabular data and text-table."""

import logging
import multiprocessing
import zipfile
import textwrap
from datetime import datetime
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import ujson as json
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, export_text
# from sklearn.ensemble import RandomForestClassifier
from eleet_pretrain.steps import Step
from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.datasets.pretraining.data_import.download import DownloadTRExStep
from eleet_pretrain.datasets.pretraining.data_import.wikidata import WikidataPreprocessStep
from eleet_pretrain.datasets.pretraining.data_import.wikidata_utils import shorten_uri
from eleet_pretrain.datasets.pretraining.mongo_processing.special_columns import ANNOTATOR, ANSWER_END, ANSWER_NORMALIZED, \
    ANSWER_START, ANSWER_SURFACEFORM, ANSWER_URI, ATTR_LABEL, ATTR_URI, DOC_INDEX, DOC_NUMBER, SUBJ_INDEX, DOC_LABEL, \
        DOC_URI, TEXT, SUBJECT_LABEL, SUBJECT_URI, HINT_OBJ, HINT_PRED_URI, HINT_PRED_LABEL

logger = logging.getLogger(__name__)


QUERY_COLS = (ATTR_URI, DOC_URI, SUBJECT_URI, ANSWER_URI, ANSWER_SURFACEFORM, ANSWER_START,
              ANSWER_END, ATTR_LABEL, DOC_LABEL, DOC_NUMBER, SUBJECT_LABEL, ANSWER_NORMALIZED, HINT_OBJ,
              HINT_PRED_URI, HINT_PRED_LABEL)

DATE_PATTERNS = [
    "%Y-%m-%dT%H:%M:%SZ^^http://www.w3.org/2001/XMLSchema#dateTime",
    "%Y-%m-%dT%H:%M:00Z^^http://www.w3.org/2001/XMLSchema#dateTime",
    "%Y-%m-%dT%H:00:00Z^^http://www.w3.org/2001/XMLSchema#dateTime",
    "%Y-%m-%dT00:00:00Z^^http://www.w3.org/2001/XMLSchema#dateTime",
    "%Y-%m-00T00:00:00Z^^http://www.w3.org/2001/XMLSchema#dateTime",
    "%Y-00-00T00:00:00Z^^http://www.w3.org/2001/XMLSchema#dateTime"
]

MODEL_FEATURES = [
    "is_matched",
    "is_date", "date_granularity_diff",
    "sentence_id_diff",
    "word_id_diff",
    "num_matched_entities"
]

DECIDER = {
    "word_id_diff": lambda x: -30 < x < 30
}


class TRExPreProcessor(BaseLoader):
    """Load TREx dataset and put in in the right format."""

    def __init__(self, dataset, dataset_dir, sample):
        """Initialize the loader."""
        super().__init__(dataset, dataset_dir, sample)
        self.prepared_text = list()
        self.prepared_queries = list()
        self.dataset_path = next(x for x in self.trex_dir.iterdir() if x.suffix == ".zip")
        self.checkpoint_id = 0
        self.model = DecisionTreeClassifier(max_depth=3)
        self.model_data_X = list()
        self.model_data_y = list()
        self.cached_answers = dict()
        self.cached_answers_path = Path(BaseLoader.get_output_dir(dataset_dir) / "cached_answers")
        self.train_model = False
        self.debug_output = False
        if self.cached_answers_path.exists():
            with open(self.cached_answers_path, "r") as f:
                self.cached_answers = json.load(f)

    def load(self, num_workers):
        """Load the dataset."""
        self.labels
        self.multiprocessing_preprocess(
            data_loader_process=self.distribute_files,
            writer_process=self.update_progress_bar,
            worker_process=self.read_files,
            num_workers=num_workers,
            writer_args=(num_workers, ),
        )
        p = multiprocessing.Process(target=self.wrap_up, daemon=True)
        p.start()
        p.join()

    def wrap_up(self):
        self.mongo_connect()
        self.query_store.create_indexes()
        self.text_store.create_indexes()

    def distribute_files(self, job_queue, num_workers):
        with zipfile.ZipFile(self.dataset_path, "r") as zip_ref:
            for name in zip_ref.namelist():
                job_queue.put(name)
        for _ in range(num_workers):
            job_queue.put(None)

    def update_progress_bar(self, num_workers, example_queue):
        sample_file_data = dict()
        with zipfile.ZipFile(self.dataset_path, "r") as zip_ref:
            total = len(zip_ref.namelist())
            with tqdm(total=total, desc="Process TREx", position=num_workers) as pbar:
                job = example_queue.get()
                while job is not None:
                    pbar.update(job[0])
                    if job[1] is not None:
                        sample_file_data = self.merge_sample_file_data(sample_file_data, job[1])
                    job = example_queue.get()
        self.generate_sample_files(sample_file_data)

    def read_files(self, job_queue, example_queue, worker_id):
        """Read all files in a directory and return for each the text table and the queries."""
        self.mongo_connect()
        job = job_queue.get()
        i = 0
        with zipfile.ZipFile(self.dataset_path, "r") as zip_ref:
            while job is not None:
                try:
                    self._read_file(zip_ref, job, worker_id)
                except Exception as e:
                    logger.warn(e, stack_info=True)

                if (i + 1) % 3 == 0:
                    self._checkpoint()
                example_queue.put((1, None))
                job = job_queue.get()
                i += 1
        example_queue.put((0, self.get_sample_file_data("labels")))
        self._checkpoint()

    def _checkpoint(self):
        self.mongo_connect()
        if len(self.prepared_queries) > 0:
            self.query_store.insert_many(pd.DataFrame(self.prepared_queries).drop_duplicates().to_dict("records"))
        if len(self.prepared_text) > 0:
            self.text_store.insert_many(pd.DataFrame(self.prepared_text, columns=[TEXT, DOC_INDEX]).to_dict("records"))

        self.prepared_queries = list()
        self.prepared_text = list()

        self.checkpoint_id += 1
        logger.info("Checkpoint!")

    def _read_file(self, zip_ref, file, worker_id):
        """Read the TREx dataset."""
        logger.info(f'Reading TREx datafile {file} into Pandas Dataframe.')
        with zip_ref.open(file) as f:
            for record in tqdm(json.load(f), desc=f"Loading {file}", position=worker_id):
                self._prepare_extract_queries(record['docid'], record)
                self.prepared_text.append({TEXT: record['text'], DOC_INDEX: shorten_uri(record["docid"])})
                self.extract_qualifiers(record)

    def extract_qualifiers(self, record):
        """Extract qualifiers from record."""

        s = shorten_uri(record["docid"])

        if s not in self.properties:
            return
        TRExPreProcessor.compute_matched_entities(record)

        triples = dict()
        for triple in record["triples"]:
            p = shorten_uri(triple["predicate"]["uri"])
            o = triple["object"]["uri"]
            triples[p] = triples.get(p, dict())
            triples[p][o] = triples[p].get(o, list())
            triples[p][o].append(triple)

        for p in set(self.properties[s]) & set(triples):
            q = dict(self.properties[s][p].iter("uri", True))
            for o in set(q) & set(triples[p]):
                self._match_entities_to_qualifiers(
                    record=record,
                    qualifiers=q[o],
                    triple_mentions=triples[p][o])

    def _match_entities_to_qualifiers(self, record, qualifiers, triple_mentions):
        """Match qualifiers with linked entities."""
        qualifier_mentions = self.get_qualifier_mentions(record, qualifiers, triple_mentions)

        for q_pred, mentions in qualifier_mentions.items():
            if mentions:
                if self.debug_output or self.train_model:
                    self.print_qualifier_match(text=record["text"],
                                               q_pred=q_pred,
                                               q_obj_mentions=mentions)
                self.prepare_extract_qualifiers(docid=record["docid"],
                                                q_pred=q_pred,
                                                q_obj_mentions=mentions)

    def get_qualifier_mentions(self, record, qualifiers, triple_mentions):
        """Compute which entities might represent a mention of a qualifier."""
        entities = dict()
        for entity in record["entities"]:
            uri = entity["uri"]
            entities[uri] = entities.get(uri, list())
            entities[uri].append(entity)

        qualifier_mentions = dict()  # collect linked entities, that could be a mention of the qualifier.
        for q_pred, q_objs in qualifiers.items():
            qualifier_mentions[q_pred] = dict()
            for q_obj in q_objs.iter("uri"):
                match_func = self.match_date if q_obj.endswith("#dateTime") else self.match_entity
                match_func(record=record, q_obj=q_obj, entities=entities, qualifier_mentions=qualifier_mentions,
                        q_pred=q_pred, triple_mentions=triple_mentions)
        return qualifier_mentions

    def match_entity(self, record, q_obj, entities, qualifier_mentions, q_pred, triple_mentions):
        """Match the qualifier with a matched wikidata entity."""
        if q_obj in entities:
            qualifier_mentions[q_pred][q_obj] = qualifier_mentions[q_pred].get(q_obj, list())
            qualifier_mentions[q_pred][q_obj].extend(
                self.compute_mention_features(record, entities[q_obj], triple_mentions)
            )

    def match_date(self, record, q_obj, entities, qualifier_mentions, q_pred, triple_mentions):
        """Match the date q_obj wth the matched entities using different date patterns."""
        t = None
        wikidata_date_granularity = None
        for i, pattern in enumerate(DATE_PATTERNS[::-1]):
            try:
                t = datetime.strptime(q_obj[1:], pattern)
                wikidata_date_granularity = i
                break
            except ValueError:
                pass

        if t is None:
            logger.warning(f"Unable to parse date: {q_obj}.")
            return

        added = set()
        for i, pattern in enumerate(DATE_PATTERNS[::-1]):
            patternised = t.strftime(pattern)
            if patternised not in added and t.strftime(pattern) in entities:
                qualifier_mentions[q_pred][q_obj] = qualifier_mentions[q_pred].get(q_obj, list())
                qualifier_mentions[q_pred][q_obj].extend(
                    self.compute_mention_features(record, entities[t.strftime(pattern)],
                                                  triple_mentions, wikidata_date_granularity, i)
                )
                added.add(patternised)

    def compute_mention_features(self, record, entities, triple_mentions,
                                 wikidata_date_granularity=-1, text_date_granularity=-1):
        """Compute features for possible the mentions of Wikidata qualifiers."""
        for entity in entities:
            entity = copy(entity)
            entity["sentence_id"] = TRExPreProcessor.compute_boundary_id(record["sentences_boundaries"], entity)
            entity["word_id"] = TRExPreProcessor.compute_boundary_id(record["words_boundaries"], entity)
            entity["is_matched"] = TRExPreProcessor.is_matched(record, entity)
            entity["is_date"] = wikidata_date_granularity >= 1
            entity["wikidata_date_granularity"] = wikidata_date_granularity
            entity["text_date_granularity"] = text_date_granularity
            entity["date_granularity_diff"] = wikidata_date_granularity - text_date_granularity
            entity["sentence_id_diff"] = float("inf")
            entity["word_id_diff"] = float("inf")
            entity["num_matched_entities"] = len(entities)

            best_triple_mention = None
            for triple_mention in triple_mentions:
                sentence_id_diff = entity["sentence_id"] - triple_mention["sentence_id"]
                word_id_diff = entity["word_id"] - TRExPreProcessor.compute_boundary_id(record["words_boundaries"],
                                                                                        triple_mention["object"])
                if (abs(sentence_id_diff), abs(word_id_diff)) \
                        < (abs(entity["sentence_id_diff"]), abs(entity["word_id_diff"])):
                    entity["sentence_id_diff"] = sentence_id_diff
                    entity["word_id_diff"] = word_id_diff
                    best_triple_mention = triple_mention
            entity["triple_mention"] = best_triple_mention
            entity["id"] = "-".join(map(str, [
                shorten_uri(record["docid"]), shorten_uri(entity["uri"]), entity["boundaries"],
                *[shorten_uri(best_triple_mention[x]["uri"]) for x in ["subject", "predicate", "object"]],
                *[best_triple_mention[x]["boundaries"] for x in ["subject", "predicate", "object"]]
            ]))
            if self.train_model or all(func(entity[key]) for key, func in DECIDER.items()):
                yield entity

    @staticmethod
    def compute_boundary_id(boundaries, entity):
        """Compute the word or sentence id using word or sentence boundaries."""
        start_entity, end_entity = entity["boundaries"]
        i, j = 0, len(boundaries)
        while i < j:
            x = int(i / 2 + j / 2)
            start_sentence, end_sentence = sorted(boundaries[x])
            if start_sentence <= start_entity and end_sentence >= end_entity:
                return x
            if end_sentence <= start_entity:
                i = x + 1
            elif start_sentence >= end_entity:
                j = x
            elif start_entity > start_sentence:
                i = x + 1
            elif start_entity < start_sentence:
                j = x
            else:
                return x
        return 0

    @staticmethod
    def compute_matched_entities(record):
        """Compute which entities are already matched to triples by TREx."""
        matched_entities = sorted(
            (t["sentence_id"], [t[x]["uri"] for x in ("subject", "predicate", "object")])
            for t in record["triples"]
        )
        record["matched_entities"] = dict()
        for sentence_id, uris in matched_entities:
            record["matched_entities"][sentence_id] = record["matched_entities"].get(sentence_id, [])
            record["matched_entities"][sentence_id].extend(uris)

    @staticmethod
    def is_matched(record, entity):
        """Check whether the entity has been already matched by TREx"""
        return entity["sentence_id"] in record["matched_entities"] \
            and entity["uri"] in record["matched_entities"][entity["sentence_id"]]

    def print_qualifier_match(self, text, q_pred, q_obj_mentions):
        """Print the matches of qualifiers for debugging purposes."""
        for q_obj, m in q_obj_mentions.items():
            for q_obj_mention in m:

                print("")
                t = q_obj_mention["triple_mention"]
                print(self.labels.get(shorten_uri(t["subject"]["uri"]),
                      t["subject"]["uri"]), end=" - ")
                print(self.labels.get(shorten_uri(t["predicate"]["uri"]),
                      t["predicate"]["uri"]), end=" - ")
                print(self.labels.get(shorten_uri(t["object"]["uri"]),
                                      shorten_uri(t["object"]["uri"])))
                print(self.labels.get(q_pred, q_pred), end=" - ")
                print(self.labels.get(shorten_uri(q_obj), shorten_uri(q_obj)))

                mentions = [tuple(q_obj_mention["boundaries"]) + ("q",)]
                for x in ["subject", "predicate", "object"]:
                    triple_mention = q_obj_mention["triple_mention"]
                    if triple_mention[x]["boundaries"]:
                        mentions.append(tuple(triple_mention[x]["boundaries"]) + (x[:1],))
                mentions = sorted(mentions)

                last_e = 0
                print("")
                for s, e, mention_type in mentions:
                    if s < last_e:
                        continue
                    for subtext in textwrap.wrap(text[last_e:s], 150):
                        print(" |", subtext.strip())
                    print(mention_type + ":",  text[s:e])
                    last_e = e
                for subtext in textwrap.wrap(text[last_e:], 150):
                    print(" |", subtext.strip())
                if self.train_model:
                    self.add_to_model(q_obj_mention)

    def add_to_model(self, entity):
        """Add a possible mention of a qualifier to a simple classification model, to come up with heuristics."""
        label = self.cached_answers.get(entity["id"], None)
        if label is None:
            label = input("Alignment correct? (y/n) > ").lower().startswith("y")
        print("Using label:", label)
        self.cached_answers[entity["id"]] = label

        self.model_data_X.append(np.array([entity[key] for key in MODEL_FEATURES]))
        self.model_data_y.append(label)
        self.model.fit(np.array(self.model_data_X), np.array(self.model_data_y))
        print(list(zip(MODEL_FEATURES, self.model.feature_importances_)))
        if hasattr(self.model, "tree_"):
            print(export_text(self.model))
        else:
            for m in self.model.estimators_:  # pylint: disable=no-member
                print(export_text(m))
        with open(self.cached_answers_path, "w") as f:
            json.dump(self.cached_answers, f)

    def prepare_extract_qualifiers(self, docid, q_pred, q_obj_mentions):
        """Print the matches of qualifiers for debugging purposes."""
        for q_obj, mentions in q_obj_mentions.items():
            for q_obj_mention in mentions:
                triple = q_obj_mention["triple_mention"]
                s, q_subj = tuple(triple[x]['uri'] for x in ('subject', 'object'))
                doc_short = shorten_uri(docid)
                q_subj_short = shorten_uri(q_subj)
                q_obj_short = shorten_uri(q_obj)

                if s != docid:
                    continue

                self.prepared_queries.append({
                    ATTR_URI: "http://www.wikidata.org/prop/direct/" + q_pred,
                    DOC_URI: docid,
                    DOC_LABEL: self.labels.get(doc_short, doc_short),
                    DOC_NUMBER: 0,
                    SUBJECT_URI: q_subj,
                    SUBJECT_LABEL: self.labels.get(q_subj_short, q_subj_short),
                    ATTR_LABEL: self.labels.get(q_pred, q_pred),
                    ANSWER_URI: q_obj,
                    ANSWER_NORMALIZED: self.labels.get(q_obj_short, q_obj_short),
                    ANSWER_SURFACEFORM: q_obj_mention['surfaceform'],
                    ANSWER_START: q_obj_mention['boundaries'][0],
                    ANSWER_END: q_obj_mention['boundaries'][1],
                    DOC_INDEX: doc_short,
                    SUBJ_INDEX: q_subj_short,
                    ANNOTATOR: 'Qualifier-Aligner'
                })

    def _prepare_extract_queries(self, docid, record):
        """Prepare the extraction of queries."""
        for _, triple in enumerate(record['triples']):
            s, p, o = tuple(triple[x]['uri'] for x in ('subject', 'predicate', 'object'))
            doc_short = shorten_uri(docid)
            s_short = shorten_uri(s)
            p_short = shorten_uri(p)
            o_short = shorten_uri(o)

            # if triple["annotator"] == "Simple-Aligner":  # Skip noisy aligner TODO
            #     continue

            self.prepared_queries.append({
                ATTR_URI: p,
                DOC_URI: docid,
                DOC_LABEL: self.labels.get(doc_short, doc_short),
                DOC_NUMBER: 0,
                SUBJECT_URI: s,
                SUBJECT_LABEL: self.labels.get(s_short, s_short),
                ANSWER_URI: o,
                ATTR_LABEL: self.labels.get(p_short, p_short),
                ANSWER_NORMALIZED: self.labels.get(o_short, o_short),
                ANSWER_SURFACEFORM: triple['object']['surfaceform'],
                ANSWER_START: triple['object']['boundaries'][0],
                ANSWER_END: triple['object']['boundaries'][1],
                ANNOTATOR: triple['annotator'],
                DOC_INDEX: doc_short,
                SUBJ_INDEX: s_short,
            })
            self._add_reflexive_relation(docid, doc_short, s, s_short, triple["subject"], triple["annotator"])
            self._add_reflexive_relation(docid, doc_short, o, o_short, triple["object"], triple["annotator"])

    def _add_reflexive_relation(self, docid, doc_short, x, x_short, x_def, annotator):
        pronouns = ("he", "she", "it", "him", "her", "they", "them", "none")

        if (str(x_def['surfaceform']).lower() not in pronouns
                and "boundaries" in x_def and x_def["boundaries"] is not None):
            self.prepared_queries.append({
                ATTR_URI: "id",
                DOC_URI: docid,
                DOC_LABEL: self.labels.get(doc_short, doc_short),
                DOC_NUMBER: 0,
                SUBJECT_URI: x,
                SUBJECT_LABEL: self.labels.get(x_short, x_short),
                ANSWER_URI: x,
                ATTR_LABEL: "id",
                ANSWER_NORMALIZED: self.labels.get(x_short, x_short),
                ANSWER_SURFACEFORM: x_def['surfaceform'],
                ANSWER_START: x_def['boundaries'][0],
                ANSWER_END: x_def['boundaries'][1],
                ANNOTATOR: annotator,
                DOC_INDEX: doc_short,
                SUBJ_INDEX: x_short,
            })

class TRExPreprocessStep(Step):
    """Load data from wikidata dump."""

    depends_on = {DownloadTRExStep, WikidataPreprocessStep}

    def check_done(self, args, dataset):
        """Check whether the step has already been executed."""
        x = TRExPreProcessor(dataset, args.dataset_dir, args.small_sample)
        x.mongo_connect()
        return not x.query_store.is_empty() and not x.text_store.is_empty()

    def run(self, args, dataset):
        """Execute the step."""
        x = TRExPreProcessor(dataset, args.dataset_dir, args.small_sample)
        x.load(args.num_workers)
