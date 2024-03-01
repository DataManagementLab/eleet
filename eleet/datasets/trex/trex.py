from pathlib import Path
import pandas as pd
from eleet.database import Database, Table, TextCollection, TextCollectionLabels


def load_trex_legacy(db_dir, split):
    result = dict()
    for file in (Path(db_dir) / "db" / "train").iterdir():
        name = file.name.split(".")[0]
        df = pd.read_json(file)
        df["index"] = df.index - max(df.index) -1
        df = df.set_index("index")
        result["train_" + name] = df.sort_index()

    for file in (Path(db_dir) / "db" / split).iterdir():
        name = file.name.split(".")[0]
        df = pd.read_json(file)
        result[name] = df.sort_index()

    assert (result["nobel-Personal-normed"]["name"].apply(lambda x: x[0]) \
            == result["nobel-Career-normed"]["name"].apply(lambda x: x[0])).all()
    assert (result["countries-Geography-normed"]["name"].apply(lambda x: x[0]) \
            == result["countries-Politics-normed"]["name"].apply(lambda x: x[0])).all()
    assert (result["skyscrapers-History-normed"]["name"].apply(lambda x: x[0]) \
            == result["skyscrapers-Location-normed"]["name"].apply(lambda x: x[0])).all()
    assert (result["nobel-Personal-reports"] == result["nobel-Career-reports"]).all().all()
    assert (result["countries-Geography-reports"] == result["countries-Politics-reports"]).all().all()
    assert (result["skyscrapers-History-reports"] == result["skyscrapers-History-reports"]).all().all()

    table_names = ("nobel-Personal", "nobel-Career", "countries-Geography",
                   "countries-Politics", "skyscrapers-Location", "skyscrapers-History")
    tables = [
        *[Table(name=f"{t}-union", data=result[f"train_{t}-labels"].reset_index(), key_columns=["index"])
          for t in table_names],
        *[Table(name=f"{t}-join", data=result[f"{t}-labels"].reset_index(), key_columns=["index"])
          for t in table_names]
    ]

    text_collections = [
        TextCollection(
            name=f"{t.split('-')[0]}_reports",
            data=result[f"{t}-reports"].reset_index(),
            key_columns=["index"]
        ) for t in ("nobel-Personal", "countries-Geography", "skyscrapers-Location")
    ]

    for text_collection in text_collections:
        for t in table_names:
            if t.startswith(text_collection.name.split("_")[0]):
                text_collection.setup_text_table(
                        table_name=t.split("-")[1],
                        attributes=result[f"{t}-labels"].columns.tolist(),
                        multi_row=False,
                        labels=TextCollectionLabels(
                            normed=result[f"{t}-normed"].reset_index().set_index("index"),
                            alignments=result[f"{t}-alignment"].reset_index().set_index("index")
                        )
                )

    database = Database(name="trex", tables=tables, texts=text_collections)
    return database


if __name__ == "__main__":
    db = load_trex_legacy("datasets/trex", "train")
    print(db)