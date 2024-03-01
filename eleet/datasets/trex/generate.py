from pathlib import Path
import logging
import argparse
from eleet.datasets.trex.generate_dataset import TRExExtractor


logging.basicConfig(format="%(levelname)s %(asctime)s %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = (Path(__file__).parents[3] / "datasets" / "trex" / ".cache").absolute()
DATA_DIR = (Path(__file__).parents[3] / "datasets" / "trex" / "data").absolute()
FINAL_FILE_NAME = "trex_new.h5"
TRAINING_SUBSET_DIR_NAME = "training_subsets_new"

CACHE_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).parents[3] / "datasets",
                        help="Root directory of datasets.")
    args = parser.parse_args()

    extractor = TRExExtractor("trex-wikidata", args.dataset_dir , sample=False, data_dir=DATA_DIR, cache_dir=CACHE_DIR)

    # Step 1: Generate ELEET Data
    if not (CACHE_DIR.parent / "db").exists():
        extractor.generate_db(out_dir=CACHE_DIR.parent)


if __name__ == "__main__":
    main()