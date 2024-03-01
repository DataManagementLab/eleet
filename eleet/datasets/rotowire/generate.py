import zipfile
import shutil
from pathlib import Path
import gdown
import logging
from eleet.datasets.rotowire.generate_dataset import GenerateDataset
from eleet.datasets.rotowire.align import AlignerTrainer
from eleet.datasets.rotowire.wiki_infoboxes import WikiInfoboxes

logging.basicConfig(format="%(levelname)s %(asctime)s %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = (Path(__file__).parents[3] / "datasets" / "rotowire" / ".cache").absolute()
DATA_DIR = (Path(__file__).parents[3] / "datasets" / "rotowire" / "data").absolute()
FINAL_FILE_NAME = "rotowire_new.h5"
TRAINING_SUBSET_DIR_NAME = "training_subsets_new"

CACHE_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":

    # Step 1: Download
    if not DATA_DIR.exists() or next(DATA_DIR.iterdir(), None) is None:
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        gdown.cached_download("https://drive.google.com/u/0/uc?id=1zTfDFCl1nf_giX7IniY5WbXi9tAuEHDn",
                              DATA_DIR / "data-release.zip")
        with zipfile.ZipFile(DATA_DIR / "data-release.zip", 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        for f in (DATA_DIR / "data" / "rotowire").iterdir():
            f.rename(DATA_DIR / f.name)
        shutil.rmtree(DATA_DIR / "data")

    # Step 2: Compute Wikidata Rotowire mapping and Evidences
    w = WikiInfoboxes(DATA_DIR, CACHE_DIR)
    mapping_path = clf_path = Path(__file__).parent / "rotowire-wikipedia-mapping.txt"
    if not (CACHE_DIR / "rotowire-wikipedia-mapping.txt").exists():
        if mapping_path.exists():
            shutil.copy(mapping_path, CACHE_DIR / "rotowire-wikipedia-mapping.txt")
        else:
            w.compute_rotowire_wikipedia_mapping()
    if not (CACHE_DIR / "wikipedia-evidences-team.jsonl").exists():
        w.compute_evidences()

    # Step 4: Train alignment model
    clf_path = Path(__file__).parent / "train.clf"
    if not clf_path.exists():
        a = AlignerTrainer(DATA_DIR, CACHE_DIR)
        try:
            a.run_interactive_alignment_training()
        finally:
            shutil.copy((CACHE_DIR / "train.clf"), clf_path)

    # Step 6: Generate Data
    dataset_generator = GenerateDataset(DATA_DIR, CACHE_DIR)
    if not (CACHE_DIR / "test-alignment.jsonl").exists():
        dataset_generator.generate_alignments()
    if not dataset_generator.db_dir.exists():
        dataset_generator.generate_db()
