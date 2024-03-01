"""Download wikidata and TREx dataset."""

from typing import Optional
import argparse
import logging
import shutil
from pathlib import Path

import wget
from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.steps import Step

TREX_URL = "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/8760241/TREx.zip"
WIKIDATA_URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"
WIKIPEDIA_URL = "http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20210601/enwiki-20210520-pages-articles.xml.bz2"  # noqa
WEBTABLES_URL = "http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/"
KELM_URL = "https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/kelm_generated_corpus.jsonl"
KELM_ENTITIES_URL = "https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/entities.jsonl"

TREX_SAMPLE_URL = "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/8768701/TREx_json_sample.zip"
WIKIPEDIA_SAMPLE_URL = "http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20210601/enwiki-20210520-pages-articles1.xml-p1p41242.bz2"  # noqa
WEBTABLES_SAMPLE_URL = "http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/sample.gz"


logger = logging.getLogger(__name__)


class DownloadStep(Step):
    """Download the Wikidata and TREx datasets from the web."""
    name: Optional[str] = None
    url: Optional[str] = None
    sample_url: Optional[str] = None
    target_dir = lambda x: None  # noqa

    def check_done(self, args, **kwargs):
        """Check whether the step has already been executed."""
        return type(self).target_dir(args).exists()

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add the arguments of the parser."""
        parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).parents[3] / "datasets",
                            help="Root directory of datasets.")
        parser.add_argument("--small-sample", action="store_true")

    def run(self, args, **kwargs):
        """Execute the step."""
        try:
            target_dir = type(self).target_dir(args)
            logger.info(f"Downloading {self.name} dataset into {target_dir}")
            target_dir.mkdir(exist_ok=True, parents=True)
            print(f"Downloading {self.name} dataset into {target_dir}")
            wget.download(self.url if not args.small_sample else self.sample_url, str(target_dir))
        except KeyboardInterrupt as e:
            shutil.rmtree(target_dir)
            raise e


class DownloadTRExStep(DownloadStep):
    """Download the TREx dataset."""
    name = "TREx"
    url = TREX_URL
    sample_url = TREX_SAMPLE_URL
    target_dir = lambda args: BaseLoader.get_trex_dir(args.dataset_dir, args.small_sample)  # noqa


class DownloadWikidataStep(DownloadStep):
    """Download the Wikidata dataset."""
    name = "Wikidata"
    url = WIKIDATA_URL
    sample_url = WIKIDATA_URL
    target_dir = lambda args: BaseLoader.get_wikidata_dir(args.dataset_dir, args.small_sample)  # noqa
