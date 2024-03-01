"""Commandline interface to load data."""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["RAYON_RS_NUM_CPUS"] = "1"

import argparse
import logging
from eleet_pretrain.metrics.data_statistics import DataStatisticsStep
import resource
from eleet_pretrain.datasets import WikidataPreprocessStep, TRExPreprocessStep, DataCombineStep, \
    DownloadTRExStep, DownloadWikidataStep, WikidataClassHierarchyStep, \
    InputFormattingStep, TextPreparationStep, QueryPreparationStep , QueryEnrichmentStep
from eleet_pretrain.steps import run_steps


if __name__ == "__main__":
    datasets = {
        "trex-wikidata": [DownloadTRExStep(), DownloadWikidataStep(),
                          WikidataPreprocessStep(), WikidataClassHierarchyStep(), TRExPreprocessStep(),
                          QueryPreparationStep(), TextPreparationStep(), QueryEnrichmentStep(), DataCombineStep(),
                          InputFormattingStep(), DataStatisticsStep()],
    }
    help_text = {
        "trex-wikidata": "Queries generated from TREx, tables from Wikidata.",
        "trex-webtables": "Queries generated from TREx, tables from Webtables.",
        "gentext-wikidata": "Queries, texts and tables are generated from Wikidata.",
        "kelm-wikidata": "Queries and texts come from the KELM corpus, tables are generated from Wikidata."
    }

    parser = argparse.ArgumentParser(description="Load a dataset.")
    parser.add_argument("--log-level", type=lambda x: getattr(logging, x.upper()), default=logging.INFO)
    parser.add_argument("--memory-limit", type=int, default=resource.RLIM_INFINITY, help="Memory limit in GB")
    subparsers = parser.add_subparsers()
    for name, steps in datasets.items():
        subparser = subparsers.add_parser(name, help=help_text[name], conflict_handler="resolve")
        for step in steps:
            step.add_arguments(subparser)
        subparser.set_defaults(func=run_steps(steps, dataset=name))
    args = parser.parse_args()

    if args.memory_limit != resource.RLIM_INFINITY:
        soft_limit = int(args.memory_limit * (1024 ** 3))
        hard_limit = int((args.memory_limit + 1) * (1024 ** 3))
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))

    try:
        args.func(args)
    except AttributeError as e:
        if "'func'" not in str(e):
            raise e
        parser.print_help()
        print()
        print(f"Choose one of these datasets: {list(datasets.keys())}")
