import pandas as pd


class TRExCollectedData():
    def __init__(self, data, from_files=False):
        self.reports_tables = dict()
        self.alignments_tables = dict()
        self.labels_tables = dict()
        self.normed_tables = dict()

        if from_files:
            self._init_from_files(data)
            return

        for table, d in data.items():
            texts, labels_ours, labels_t2t, alignments = tuple(zip(*d))
            entity_ids = [l.name[1] for l in alignments]

            self.reports_tables[table] = pd.DataFrame(texts)
            self.reports_tables[table].columns = ["text"]
            self.reports_tables[table].index = entity_ids
            self.reports_tables[table].index.name = "ID"

            self.labels_tables[table] = pd.DataFrame(labels_t2t)
            self.labels_tables[table].reset_index(drop=True, inplace=True)
            self.labels_tables[table].index = entity_ids
            self.labels_tables[table].index.name = "ID"

            self.normed_tables[table] = pd.DataFrame(labels_ours)
            self.normed_tables[table].reset_index(drop=True, inplace=True)
            self.normed_tables[table].index = entity_ids
            self.normed_tables[table].index.name = "ID"

            self.alignments_tables[table] = pd.DataFrame(alignments)
            self.alignments_tables[table].reset_index(drop=True, inplace=True)
            self.alignments_tables[table].index = entity_ids
            self.alignments_tables[table].index.name = "ID"
            self.alignments_tables[table] = self.alignments_tables[table].applymap(lambda x: x if len(x) > 0 else "")

        sub_datasets = sorted(set([k[0] for k in data.keys()]))  # align subtables per domain --> same num roes /texts
        for dataset_name in sub_datasets:
            tables = sorted([t for t in data.keys() if t[0] == dataset_name])
            texts_merged = pd.concat((self.reports_tables[t] for t in tables), axis=1).apply(
                lambda x: max(x, key=lambda y: isinstance(y, str)), axis=1)

            for t in tables:
                self.reports_tables[t] = texts_merged
                missing = set(texts_merged.index) - set(self.labels_tables[t].index)
                for m in sorted(missing):
                    self.labels_tables[t].loc[m] = ""
                    self.alignments_tables[t].loc[m] = ""
                    self.normed_tables[t].loc[m] = ""

                    for x in set(tables) - {t}:
                        for k in self.labels_tables[x].loc[m].index:
                            if k in self.labels_tables[t].loc[m] and self.labels_tables[x].loc[m, k] != "":
                                self.labels_tables[t].loc[m, k] = self.labels_tables[x].loc[m, k]
                                self.alignments_tables[t].loc[m, k] = self.alignments_tables[x].loc[m, k]
                                self.normed_tables[t].loc[m, k] = self.normed_tables[x].loc[m, k]

                self.reports_tables[t].sort_index(inplace=True)
                self.alignments_tables[t].sort_index(inplace=True)
                self.labels_tables[t].sort_index(inplace=True)
                self.normed_tables[t].sort_index(inplace=True)

        for k in data.keys():
            self.reports_tables[k].reset_index(inplace=True, drop=True)
            self.alignments_tables[k].reset_index(inplace=True, drop=True)
            self.labels_tables[k].reset_index(inplace=True, drop=True)
            self.normed_tables[k].reset_index(inplace=True, drop=True)
            self.reports_tables[k] = pd.DataFrame(self.reports_tables[k], columns=["Report"])

        self.tables = sorted(data.keys())
        self.rm_name_not_tagged()

    @staticmethod
    def init_from_files(data_path, split):
        result = dict()
        for file in (data_path / split).iterdir():
            name = file.name.split(".")[0]
            df = pd.read_json(file)
            result[name] = df
        return TRExCollectedData(result, from_files=True)

    def _init_from_files(self, data):
        for key, value in data.items():
            k = tuple(key.split("-")[:2])
            d = {"reports": self.reports_tables, "alignment": self.alignments_tables,
                 "labels": self.labels_tables, "normed": self.normed_tables}[key.split("-")[2]]
            d[k] = value.sort_index()

        self.rm_name_not_tagged()

    def rm_name_not_tagged(self):
        for k in self.alignments_tables:
            mask = self.alignments_tables[k]["name"] != ""
            self.reports_tables[k] = self.reports_tables[k][mask].reset_index(drop=True)
            self.alignments_tables[k] = self.alignments_tables[k][mask].reset_index(drop=True)
            self.labels_tables[k] = self.labels_tables[k][mask].reset_index(drop=True)
            self.normed_tables[k] = self.normed_tables[k][mask].reset_index(drop=True)