"""Combine data loaded by the loaders."""

import logging

import numpy as np
import re
from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.datasets.pretraining.python_processing.utils import SPLITS, to_dict
from eleet_pretrain.datasets.pretraining.mongo_processing.query_enrichment import QueryEnrichmentStep
from eleet_pretrain.datasets.pretraining.mongo_processing.query_preparation import QueryPreparationStep
from eleet_pretrain.steps import Step
import multiprocessing
from eleet_pretrain.datasets.pretraining.data_import.wikidata_utils import shorten_uri
from eleet_pretrain.datasets.pretraining.mongo_processing.mongo_store import multiprocessing_get_ranges, multiprocessing_update_process_bar
from eleet_pretrain.datasets.pretraining.python_processing.testing import TestingDataPreprocessor
from eleet_pretrain.datasets.pretraining.python_processing.pretraining import PretrainingDataPreprocessor
from eleet_pretrain.datasets.pretraining.python_processing.collected_data import CollectedData


logger = logging.getLogger(__name__)


class DataCombiner(BaseLoader):
    """Merge data from the different sources."""

    def combine(self, num_workers):
        self.labels
        self.aliases
        self.store.reset()
        q = multiprocessing.Queue()
        p1 = multiprocessing.Process(target=self.compute_relevant_properties, daemon=True, args=(q,))
        for f in [p1.start, p1.join]:
            f()

        relevant_predicates = q.get()
        self.multiprocessing_preprocess(
            data_loader_process=multiprocessing_get_ranges(self.prepared_query_stores, 5 if self.sample else 100),
            writer_process=multiprocessing_update_process_bar(self.prepared_query_stores, "Tabularizing",
                                                              sample_file=True, loader=self),
            worker_process=self._combine,
            num_workers=num_workers,
            worker_args=(relevant_predicates, ),
        )

    def compute_relevant_properties(self, q):
        self.mongo_connect()
        q.put(set(shorten_uri(x) for x in self.query_store.distinct("_attr_uri")) - {"id"})


    def _combine(self, relevant_properties, job_queue, example_queue, worker_id):
        """Combine data with and Wikidata."""
        self.mongo_connect()
        job = job_queue.get()
        rng = np.random.default_rng(worker_id)
        i = 0
        while job is not None:
            skip, limit, total, split = job
            pretraining = SPLITS[split] in ("train", "development")
            preprocessor = (PretrainingDataPreprocessor if pretraining else TestingDataPreprocessor)(
                labels=self.labels, aliases=self.aliases, relevant_properties=relevant_properties, rng=rng
            )
            cursor = self.get_cursor(split, skip, limit)
            data = CollectedData(self.aliases, rng, prefix=f"{worker_id}-{i}")
            for docs_orig in cursor:
                evidences_dict, texts_dict = to_dict(docs_orig["_evidences_"], "wikidata_id"), \
                            to_dict(docs_orig["_texts_"], "text_idx")

                table_name = self.get_table_name(docs_orig)
                self.fix_missing_row_label_tags(docs_orig, texts_dict)
                preprocessor.construct_dataset(docs=docs_orig, result_data=data, evidences_dict=evidences_dict,
                                               texts_dict=texts_dict, table_name=table_name, split=SPLITS[split],
                                               do_augment=(split==0 and not self.sample))

            self.checkpoint(data=data, batch_id=f"{worker_id}-{i}", split_id=split)
            example_queue.put((limit, total, split, self.get_sample_file_data("labels", "aliases")))
            job = job_queue.get()
            i += 1

    def fix_missing_row_label_tags(self, docs_orig, texts_dict):
        for doc in docs_orig["docs"]:
            given = [(m["_answer_surfaceform"], m["_answer_start"], m["_answer_end"])
                     for q in doc["_queries"] if q["_attr_uri"] == "id"
                     for a in q["_answers"] for m in a["_answer_mentions"]]
            doc_id = doc["_id"]["_doc"]
            normalized = self.labels.get(doc_id, doc_id)
            identifiers = [normalized] + self.aliases.get(doc_id, [])
            matches = self._find_matches(texts_dict[doc_id]["text"], identifiers)
            if matches:
                matches = self._merge_row_label_tags(given, matches)
                query = {
                    "_attr_label": "id",
                    "_attr_uri": "id",
                    "_attr_description": "id",
                    "_answers": [{
                        "_answer_uri": doc["_id"]["_doc_uri"],
                        "_answer_normalized": normalized,
                        "_dependent_queries": [],
                        "_answer_mentions": [
                            {
                                "_answer_start": start,
                                "_answer_end": end,
                                "_answer_surfaceform": x
                            }
                            for x, start, end in matches
                        ]
                    }]
                }
                doc["_queries"] = [q for q in doc["_queries"] if q["_attr_uri"] != "id"] + [query]

                covers = [
                    {
                        "s": doc_id,
                        "p": "id",
                        "o": doc["_id"]["_doc_uri"],
                        "start": start,
                        "end": end
                    }
                    for _, start, end in matches
                ]

                texts_dict[doc_id]["covers"] = [c for c in texts_dict[doc_id]["covers"] if c["p"] != "id"] + covers
                doc["text"]["covers"] = [c for c in doc["text"]["covers"] if c["p"] != "id"] + covers

            # confusion
            if "confusion_idx" not in texts_dict[doc_id]:
                continue
            conf_id = texts_dict[doc_id]["confusion_idx"]
            given = [(x["start"], x["end"]) for x in texts_dict[doc_id]["confusion_covers"]
                      if x["p"] == "id" and x["s"] == conf_id]
            identifiers = [self.labels.get(conf_id, conf_id)] + self.aliases.get(conf_id, [])
            matches = self._find_matches(texts_dict[doc_id]["confusion"], identifiers, w_surfaceform=False) 
            if matches:
                matches = self._merge_row_label_tags(given, matches)
                covers = [
                    {
                        "s": conf_id,
                        "p": "id",
                        "o": "http://www.wikidata.org/entity/" + conf_id,
                        "start": start,
                        "end": end
                    }
                    for start, end in matches
                ]
                texts_dict[doc_id]["confusion_covers"] = [
                    c for c in texts_dict[doc_id]["confusion_covers"] if c["p"] != "id"] + covers

    def _find_matches(self, text, identifiers, w_surfaceform=True):
        matches = [(x, e.span()[0], e.span()[1]) if w_surfaceform else (e.span()[0], e.span()[1])
            for x in identifiers
            for e in re.finditer(r"(?<!\w)" + re.escape(x) + r"(?!\w)", text, re.IGNORECASE)
        ]
        if any(match[-2] < 20 for match in matches):
            return matches

        for identifier in identifiers:
            identifier = identifier.split()
            for i in range(1, len(identifier)):
                prefix, suffix = " ".join(identifier[:i]), " ".join(identifier[i:])
                prefix_match = min([e.span()[0]
                    for e in re.finditer(r"(?<!\w)" + re.escape(prefix) + r"(?!\w)", text[:50], re.IGNORECASE)
                ] + [float("inf")])
                suffix_match = min([e.span()[1]
                    for e in re.finditer(r"(?<!\w)" + re.escape(suffix) + r"(?!\w)", text[:50], re.IGNORECASE)
                ] + [float("inf")])
                if prefix_match < 50 and suffix_match < 50 and prefix_match < suffix_match:
                    matches.append((text[prefix_match: suffix_match], prefix_match, suffix_match) if w_surfaceform else
                                   (prefix_match, suffix_match))
        
        return matches

    def _merge_row_label_tags(self, *matches):
        matches = sorted([a for b in matches for a in b], key=lambda a: (a[-2], -a[-1]))
        result = list()
        current_end = 0
        for match in matches:
            start, end = match[-2], match[-1]
            if start >= current_end:
                result += [match]
                current_end = end
        return result

    def get_table_name(self, docs_orig):
        t = docs_orig["_id"].get("type", "")
        n = "".join(sorted(docs_orig["_id"].get("neighbor", {}).values()))
        o = "-".join(f"{k}:{v}" for k, v in sorted(docs_orig["_id"].items()) if k not in ("type", "neighbor"))
        return f"{t}/{n}-{o}" if n else f"{t}-{o}"

    def get_cursor(self, split, skip, limit):
        return self.prepared_query_stores[split].aggregate([
            {"$skip": skip}, {"$limit": limit},
            {"$set": {"_evidences_": {"$setUnion": [
                [{"$ifNull": [f"$docs._evidence", []]}]
            ]}}},
            {"$set": {"_evidences_": {"$reduce": {"input": "$_evidences_",
                                                  "initialValue": [],
                                                  "in": {"$setUnion": ["$$this", "$$value"]}}}}},
            {"$set": {"_texts_": {"$ifNull": [f"$docs.text.text_idx", ""]}}},
            {"$lookup": {
                "from": self.wikidata_store.collection_name,
                "localField": "_evidences_",
                "foreignField": "wikidata_id",
                "as": "_evidences_"
            }},
            {"$lookup": {
                "from": self.prepared_text_store.collection_name,
                "localField": "_texts_",
                "foreignField": "text_idx",
                "as": "_texts_"
            }}
        ])

    def checkpoint(self, data, batch_id, split_id):
        logger.info("Checkpoint!")

        if data.is_empty():
            return

        for d in data.datasets:
            for k, v in data.items(dataset_name=d):
                self.store[f"{SPLITS[split_id]}_{d}/{batch_id}/{k}"] = v


class DataCombineStep(Step):
    """Load data from wikidata dump."""
    depends_on = {QueryPreparationStep, QueryEnrichmentStep}

    def check_done(self, args, dataset):
        """Check whether the step has already been executed."""
        x = DataCombiner(dataset, args.dataset_dir, args.small_sample)
        return all(y in x.store.nodes() for y in ("train_default", "development_default", "unseen_query_default_join",
                                                  "nobel_default_join", "countries_default_join"))

    def run(self, args, dataset):
        """Execute the step."""
        x = DataCombiner(dataset, args.dataset_dir, args.small_sample)
        x.combine(args.num_workers)
