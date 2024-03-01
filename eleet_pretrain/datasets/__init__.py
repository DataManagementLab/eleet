"""Utilities for loading datasets."""
from eleet_pretrain.datasets.pretraining.data_import.wikidata import WikidataPreprocessStep  # noqa
from eleet_pretrain.datasets.pretraining.data_import.trex import TRExPreprocessStep  # noqa  # noqa
from eleet_pretrain.datasets.pretraining.python_processing.combine import DataCombineStep  # noqa
from eleet_pretrain.datasets.pretraining.data_import.download import DownloadTRExStep, DownloadWikidataStep  # noqa
from eleet_pretrain.datasets.input_formatting.input_tensoriser import InputFormattingStep  # noqa
from eleet_pretrain.datasets.pretraining.mongo_processing.text_preparation import TextPreparationStep  # noqa
from eleet_pretrain.datasets.pretraining.mongo_processing.query_preparation import QueryPreparationStep  # noqa
from eleet_pretrain.datasets.pretraining.mongo_processing.query_enrichment import QueryEnrichmentStep  # noqa
from eleet_pretrain.datasets.pretraining.data_import.wikidata_class_hierarchy import WikidataClassHierarchyStep  # noqa