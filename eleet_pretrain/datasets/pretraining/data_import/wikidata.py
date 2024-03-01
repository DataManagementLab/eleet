"""Load Trex-dataset into tabular data and text-table."""

from contextlib import ExitStack
import argparse
import logging
import multiprocessing
import bz2
from pymongo import UpdateOne

import ujson as json
from collections import defaultdict
from qwikidata.entity import WikidataItem, WikidataProperty
from tqdm import tqdm
from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.datasets.pretraining.data_import.download import DownloadWikidataStep
from eleet_pretrain.steps import Step

type_to_entity_class = {"item": WikidataItem, "property": WikidataProperty}
logger = logging.getLogger(__name__)

ARCHIVE_SPLITTED_DIR = "_splitted"
TOTAL_NUM_ENTITIES = 93512660


class WikidataImporter(BaseLoader):
    def __init__(self, dataset_dir, dataset, sample):
        """Initialize the loader."""
        super().__init__(dataset, dataset_dir, sample)
        self.dump_path = next(x for x in (self.wikidata_dir).iterdir() if x.is_file())
        self.splitted_path = self.wikidata_dir / ARCHIVE_SPLITTED_DIR
        self.batch_size = 10_000

    def process_dump(self, num_workers):
        dl = self.data_loader_process_read if self.splitted_path.exists() else self.data_loader_process_split
        wp = self.worker_process_read if self.splitted_path.exists() else self.worker_process_split
        qs = 1_000 if self.splitted_path.exists() else 1_000

        p = multiprocessing.Process(target=self.init_dump_processing, daemon=True)
        p.start()
        p.join()

        self.multiprocessing_preprocess(
            data_loader_process=dl,
            writer_process=self.write_labels_and_aliases,
            worker_process=wp,
            num_workers=num_workers,
            job_queue_maxsize=qs,
        )

    def init_dump_processing(self):
        self.wikidata_store.connect()
        self.wikidata_store.create_indexes()

    def write_labels_and_aliases(self, example_queue):
        pbar = None
        with ExitStack() as stack:
            labels_file, aliases_file = tuple(stack.enter_context(open(p, "w"))
                                              for p in (self.labels_path, self.aliases_path))
            job = example_queue.get()
            while job is not None:
                num, labels, aliases = job
                print(labels, file=labels_file)
                print(aliases, file=aliases_file)
                if num is not None and pbar is None:
                    pbar = stack.enter_context(tqdm(total=TOTAL_NUM_ENTITIES, desc="Process Wikidata"))
                if num is not None:
                    pbar.update(num)
                job = example_queue.get()

    def data_loader_process_split(self, job_queue, num_workers):
        self.splitted_path.mkdir()
        buffer = []
        with bz2.open(self.dump_path, "rb") as f:
            for i, line in tqdm(enumerate(f), total=TOTAL_NUM_ENTITIES, desc="Process Wikidata"):
                buffer.append(line)

                if i % self.batch_size == self. batch_size - 1:
                    job_queue.put(buffer)
                    buffer = []
        if buffer:
            job_queue.put(buffer)
        for _ in range(num_workers):
            job_queue.put(None)

    def data_loader_process_read(self, job_queue, num_workers):
        for elem in self.splitted_path.iterdir():
            job_queue.put(elem)
        for _ in range(num_workers):
            job_queue.put(None)

    def worker_process_split(self, job_queue, example_queue, worker_id):
        job = job_queue.get()
        i = 0
        while job is not None:
            path = self.splitted_path / f"wikidata_batch_{worker_id}_{i}.bz2"
            with bz2.open(path, "wb") as f:
                j = 0
                while job is not None and j < 100:
                    for line in job:
                        f.write(line)
                    labels, aliases = self.process(job)
                    example_queue.put((None, json.dumps(labels), json.dumps(aliases)))
                    job = job_queue.get()
                    j += 1
            i += 1

    def worker_process_read(self, job_queue, example_queue, worker_id):
        job = job_queue.get()
        while job is not None:
            with bz2.open(job, "rb") as f:
                while True:
                    labels, aliases = self.process((line for _, line in zip(range(self.batch_size), f)))
                    num = len(labels)
                    example_queue.put((num, json.dumps(labels), json.dumps(aliases)))
                    if num == 0:
                        break
            job = job_queue.get()

    def process(self, lines):
        self.mongo_connect()
        batch, labels, aliases = list(), dict(), dict()
        for i, l in enumerate(lines):
            line = l[:-(1 + (l[-2:] in (b",\n", b"{\n", b"[\n")))]
            if line == b"":
                continue
            try:
                entity_dict = json.loads(line)
            except ValueError:
                logger.warn(f"Could not decode {l}")
                continue
            doc, entity_id, label, alias, neighbor = self.process_entity_dict(entity_dict)
            batch.append(doc)
            labels[entity_id] = label
            if alias:
                aliases[entity_id] = alias
        if batch:
            self.insert_many(batch)
        return labels, aliases

    def insert_many(self, batch):
        insert_data_updates = [
            UpdateOne(
                {"wikidata_id": doc["wikidata_id"]},
                {"$set": doc},
                upsert=True,
            )
            for doc in batch
        ]
        self.wikidata_store.bulk_write(insert_data_updates)

    def process_entity_dict(self, entity_dict):
        entity_type = entity_dict["type"]
        entity = type_to_entity_class[entity_type](entity_dict)
        properties, neighbors = WikidataImporter._get_properties(entity)
        doc = {
            "wikidata_id": entity.entity_id,
            "aliases": entity.get_aliases(),
            "label": entity.get_label(),
            "description": entity.get_description(),
            "types": list(WikidataImporter._get_types(entity)),
            "superclasses": list(WikidataImporter._get_superclasses(entity)),
            "properties": properties,
            "neighbors": neighbors,

        }
        return doc, entity.entity_id, entity.get_label(), entity.get_aliases(), neighbors

    @staticmethod
    def _get_types(entity):
        """Check the type of the entity and add to rows."""
        for claim in entity.get_truthy_claim_group("P31"):
            if claim.mainsnak.datavalue is None or claim.mainsnak.snak_datatype != 'wikibase-item':
                continue
            yield claim.mainsnak.datavalue.value['id']

    @staticmethod
    def _get_superclasses(entity):
        for claim in entity.get_truthy_claim_group("P279"):
            if claim.mainsnak.datavalue is None or claim.mainsnak.snak_datatype != 'wikibase-item':
                continue
            yield claim.mainsnak.datavalue.value['id']

    @staticmethod
    def _get_properties(entity):
        properties = defaultdict(list)
        neighbors = []
        for cg_name, claims in entity.get_truthy_claim_groups().items():
            for claim in claims:
                properties[cg_name].append((
                    (claim.mainsnak.datavalue.datatype, claim.mainsnak.datavalue.value)
                     if claim.mainsnak.datavalue is not None else None,
                    {k: [((q.snak.datavalue.datatype, q.snak.datavalue.value)
                          if q.snak.datavalue is not None else None) for q in v]
                     for k, v in claim.qualifiers.items()} if claim.qualifiers is not None else None,
                ))
                if claim.mainsnak.datavalue is not None and claim.mainsnak.datavalue.datatype == "wikibase-entityid":
                    neighbors.append(
                        {"o": claim.mainsnak.datavalue.value["id"], "p": cg_name}
                    )
        return properties, neighbors


class WikidataPreprocessStep(Step):
    """Load data from wikidata dump."""

    depends_on = {DownloadWikidataStep}

    def check_done(self, args, dataset, **kwargs):
        """Check whether the step has already been executed."""
        x = WikidataImporter(dataset_dir=args.dataset_dir,
                             sample=args.small_sample,
                             dataset=dataset)
        x.mongo_connect()
        return not x.wikidata_store.is_empty()

    def add_arguments(self, parser: argparse.ArgumentParser):   # pylint: disable=R0801
        parser.add_argument('--num-workers', type=int, default=int(multiprocessing.cpu_count() / 2) - 1,
                            required=False)

    def run(self, args, dataset):
        """Execute the step."""
        x = WikidataImporter(dataset_dir=args.dataset_dir,
                             sample=args.small_sample,
                             dataset=dataset)
        x.process_dump(args.num_workers)
