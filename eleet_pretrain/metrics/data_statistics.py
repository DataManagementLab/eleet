"""Combine data loaded by the loaders."""

from collections import namedtuple
import os
import logging
from functools import reduce
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.datasets.pretraining.python_processing.combine import DataCombineStep
from eleet_pretrain.steps import Step
from eleet_pretrain.datasets.pretraining.python_processing.collected_data import OPERATOR_DICT
from eleet_pretrain.model.training_utils import get_counts, AGG_LEVELS

logger = logging.getLogger(__name__)

X_TICK_LABELS ={
    "db_operator": {v: k for k, v in OPERATOR_DICT.items()}
}

"""
Table-level stats:
- num values in cell --> done ?
"""

STATS_FILE_SUFFIX = ".stats.h5"

def get_stats_path(path):
    assert str(path).endswith(".h5")
    return Path(str(path)[:-3] + STATS_FILE_SUFFIX)

class DataStatistics(BaseLoader):
    """Class to compute some statistics of the preprocessed dataset."""

    def __init__(self, dataset, dataset_dir, sample, preprocessed_dataset):
        super().__init__(dataset, dataset_dir, sample)
        self.dataset_dir = dataset_dir
        self.preprocessed_dataset = preprocessed_dataset

    def run(self, num_workers):
        """Run the computation of statistics."""
        ending = self.preprocessed_dataset.name.split(".")[-1]
        for file_path in (f for f in self.preprocessed_dataset.parent.iterdir()
                          if f.name.split(".")[-1] == ending and not f.name.endswith(STATS_FILE_SUFFIX)):
            stats_path = get_stats_path(file_path)
            if stats_path.exists():
                stats_path.unlink()
            with h5py.File(file_path, "r") as file:
                for group in file:
                    with TemporaryDirectory(dir=file_path.parent) as tempdir:
                        do_export_raw = group not in ("train_default", "development_default")
                        self.multiprocessing_preprocess(
                            data_loader_process=self.data_loader_process,
                            writer_process=self.writer_process,
                            worker_process=self.worker_process,
                            num_workers=num_workers,
                            writer_args=(stats_path, group, do_export_raw),
                            loader_args=(file_path, group),
                            worker_args=(file_path, tempdir, do_export_raw),
                            num_error_retries=0 if self.sample else 20
                        )
        self.visualize_counts(stats_path)

    def data_loader_process(self, file_path, group, job_queue, num_workers):
        with h5py.File(file_path, "r") as file:
            origin = DataStatistics.get_origin(file[group])
        node_names = origin.index.unique()
        for node_name in node_names:
            sample_ids = origin.loc[node_name]["sample_id"]
            sample_ids = sample_ids.values if isinstance(sample_ids, pd.Series) else np.array([sample_ids])
            mask = sample_ids[1:] != sample_ids[:-1] + 1
            sample_ids = np.vstack((np.hstack((sample_ids[0:1], sample_ids[1:][mask])),  # mostly consecutive
                                    1 + np.hstack((sample_ids[:-1][mask], sample_ids[-1:])))).T
            job_queue.put((group, node_name, sample_ids, len(node_names)))
        for _ in range(num_workers):
            job_queue.put(None)

    def writer_process(self, stats_path, group, do_export_raw, example_queue):
        counts = {l: dict() for l in AGG_LEVELS}
        desc_suffix = " [and export raw stats]" if do_export_raw else ""
        with tqdm(desc=f"Compute statistics on {group} split" + desc_suffix) as pbar:
            job = example_queue.get()
            num_nodes = job[-1]
            pbar.total = num_nodes * len(GATHER_FUNCS)
            while job is not None:
                pbar.update(1)
                job_counts, tmpfile = job[0:2]
                for agg_level, c in job_counts:
                    key = (c.index.unique(level=0)[0], c.index.unique(level=1)[0])
                    if key not in counts[agg_level]:
                        counts[agg_level][key] = c 
                    else:
                        counts[agg_level][key] = counts[agg_level][key].add(c, fill_value=0)
                if do_export_raw:
                    self.merge_non_aggregated(stats_path, tmpfile)
                job = example_queue.get()
        self.export(counts, stats_path)

    def worker_process(self, file_path, tempdir, do_export_raw, job_queue, example_queue, worker_id):
        with h5py.File(file_path, "r") as file:
            i = 0
            job = job_queue.get()
            while job is not None:
                group, node_name, sample_ids, num_nodes = job
                sample_ids = np.hstack([np.arange(*r) for r in sample_ids])

                origin = self.get_origin(file[group], sample_ids=sample_ids)
                node_tensors = self.get_node_tensors(file[group], origin)

                for gather_func_name, gather_func, description in GATHER_FUNCS:
                    if gather_func_name != "#windows":
                        frame = self.get_statistics(node_tensors, gather_func, origin, node_name)
                    else:
                        frame = self.get_num_windows(origin)
                    counts = self.get_counts(frame, group)

                    tmpfile = None
                    if do_export_raw:
                        tmpfile = self.export_raw(tempdir, worker_id, i, frame, group)

                    example_queue.put((counts, tmpfile, num_nodes))
                    i += 1
                job = job_queue.get()

    def get_statistics(self, node_tensors, gather_func, origin, node_name):
        origin = origin.set_index(["sample_id", "table_id"])

        dataset = gather_func(self.store.get_node(node_name), node_tensors, origin)
        dataset = self.remove_empty_rows(node_tensors, origin, dataset)
        reset_index = [n for n in dataset.index.names if n not in ("table_id", "sample_id", None)]
        if reset_index:
            dataset.reset_index(reset_index, inplace=True)
        join_col = dataset.index.names[0]
        origin.reset_index("sample_id" if join_col != "sample_id" else "table_id", inplace=True)
        return origin.merge(dataset, on=[join_col]).reset_index()

    def remove_empty_rows(self, node_tensors, origin, dataset):
        if "row_id" not in dataset.index.names:
            return dataset
        k = "sample_id"
        if k not in dataset.index.names:
            k = "table_id"
        i, filled_row_id = np.where(node_tensors["input_ids"][:, :, 0] != 0)
        filled_k_id = origin.index.get_level_values(k)[i]
        old_order = list(dataset.index.names)
        other_levels = [x for x in old_order if x not in (k, "row_id")]
        dataset.reset_index(other_levels, inplace=True)
        dataset.sort_index(inplace=True)
        dataset = dataset.reorder_levels([k, "row_id"])
        indexer = np.unique(np.stack((filled_k_id, filled_row_id)).T.astype("<U22"), axis=0)
        indexer = [(str(x[0]) if k == "table_id" else int(x[0]), int(x[1])) for x in indexer]
        indexer = [t for t in indexer if t in dataset.index]
        dataset = dataset.loc[indexer]
        dataset.set_index(other_levels, append=True, inplace=True)
        dataset = dataset.reorder_levels(old_order)
        return dataset

    def get_node_tensors(self, tensors, origin):
        node_sample_ids = np.array(origin["sample_id"])
        node_tensors = {k: v[node_sample_ids] for k, v in tensors.items() if "statistics" not in k}
        return node_tensors

    def sorted(self, df):
        if not isinstance(df.index[0][-1], str):
            return None
        sorted_df =  df.sort_values(
            ["group", "counts"],
            key=(lambda x: x.map({g: i for i, g in enumerate(sorted(df.index.unique("group")))})
                if x.name == "group" else -x),
        )
        unique_df = sorted_df.reset_index()["value"].unique()
        return dict(zip(unique_df, range(len(unique_df))))

    @staticmethod
    def get_example_tbl_stats(key, col_id, col_name):
        def f(store, tensors, origin):
            df = pd.DataFrame(tensors[key][:, col_id], columns=[col_name])
            df.index = origin.index
            return df
        return f

    @staticmethod
    def get_example_row_stats(col):
        def f(store, tensors, origin):
            data = np.array(tensors[col]).reshape(-1)
            i = np.repeat(np.arange(tensors[col].shape[0]), tensors[col].shape[1])
            sample_id = origin.index.get_level_values("sample_id")[i]
            table_id = origin.index.get_level_values("table_id")[i]
            row_id = np.tile(np.arange(tensors[col].shape[1]), tensors[col].shape[0])
            df = pd.DataFrame(data, columns=[col],
                              index=pd.MultiIndex.from_arrays([sample_id, table_id, row_id],
                                                              names=["sample_id", "table_id", "row_id"]))
            return df
        return f

    @staticmethod
    def get_tbl_stats(col):
        """Load meta information containing dataset statistics and join it with the origin table."""
        def f(store, tensors, origin):
            dataset = store["header_meta"][[col]]
            dataset.index.name = "table_id"
            return dataset
        return f

    @staticmethod
    def get_num_cell_tokens(store, tensors, origin):
        num_examples, num_rows, num_cols = tensors["num_cell_tokens"].shape
        i = np.repeat(np.arange(num_examples), num_rows)
        sample_id = origin.index.get_level_values("sample_id")[i]
        row_id = np.tile(np.arange(num_rows), num_examples)
        reshaped = np.array(tensors["num_cell_tokens"]).reshape(-1, num_cols)
        df = pd.DataFrame(reshaped,
                          index=pd.MultiIndex.from_arrays((sample_id, row_id)))
        df = pd.DataFrame(df.stack(), columns=["num_cell_tokens"])
        df.index.names = ["sample_id", "row_id", "col_id"]
        return df
    
    @staticmethod
    def get_header_column_ids(store, tensors, origin):
        df = pd.DataFrame(store["header_column_ids"].stack(), columns=["header_column_ids"])
        df.index.names = ["table_id", "col_id"]
        df.index = df.index.set_levels(df.index.levels[-1].astype(int), level=-1)
        return df

    @staticmethod
    def get_row_ids(store, tensors, origin):
        df = pd.DataFrame(store["rows"][[]].reset_index("row_id").stack(), columns=["row_ids"])
        df.index.names = ["table_id", "row_id"]
        DataStatistics._to_numeric_row_id(df)
        return df

    @staticmethod
    def get_table_types(store, tensors, origin):
        df = store["header_meta"]["table_name"]
        df = df.apply(lambda x: str(x).split("-")[0])
        df.index.name = "table_id"
        df.name = "table_types"
        df = pd.DataFrame(df)
        return df

    @staticmethod
    def get_num_windows(origin):
        df = origin.copy()
        df["num_windows"] = 1
        del df["table_name"]
        return df.reset_index(drop=True)

    @staticmethod
    def get_queried_attrs(store, tensors, origin):
        e_id, r_id, t_id = np.where((tensors["input_ids"] == 103) & (tensors["token_type_ids"] == 1))
        c_id = tensors["column_token_position_to_column_ids"][e_id, r_id, t_id]
        table_ids = origin.index.get_level_values("table_id")[e_id]
        indexer = list(zip(table_ids, c_id.astype(str)))
        s = store["header_column_ids"].stack()[indexer]
        df = pd.DataFrame(s, columns=["queried_attrs"])
        df["row_id"] = r_id
        df.index.names = ["table_id", "col_id"]
        df = df.set_index("row_id", append=True).reorder_levels(["table_id", "row_id", "col_id"])
        df.index = df.index.set_levels(df.index.levels[-1].astype(int), level=-1)
        return df

    @staticmethod
    def get_num_answers(store, tensors, origin):
        # start with construction a df with zeros ==> don't miss zero answer queries
        ez_id, rz_id, tz_id = np.where((tensors["input_ids"] == 103) & (tensors["token_type_ids"] == 1))
        sz_id = np.array(origin.index.get_level_values("sample_id")[ez_id])
        cz_id = tensors["column_token_position_to_column_ids"][ez_id, rz_id, tz_id]
        stacked = np.hstack((sz_id.reshape(-1, 1), rz_id.reshape(-1, 1), cz_id.reshape(-1, 1)))
        df_zero = pd.DataFrame(stacked, columns=["sample_id", "row_id", "col_id"])
        df_zero["num_answers"] = 0
        df_zero.set_index(["sample_id", "row_id", "col_id"], inplace=True)

        # for all queries that have answers, compute number of distinct answers
        e_id, r_id, a_id = np.where(tensors["answer_col_ids"] > 0)
        s_id = np.array(origin.index.get_level_values("sample_id")[e_id])
        c_id = tensors["answer_col_ids"][e_id, r_id, a_id]
        answers = tensors["normalized_answers"][e_id, r_id, a_id]
        stacked = np.hstack((s_id.reshape(-1, 1), r_id.reshape(-1, 1), c_id.reshape(-1, 1), answers))
        unique_answers = np.unique(stacked, axis=0)
        unique, counts = np.unique(unique_answers[:, :3], axis=0, return_counts=3)
        df = pd.DataFrame(unique, columns=["sample_id", "row_id", "col_id"])
        df["num_answers"] = counts
        df.set_index(["sample_id", "row_id", "col_id"], inplace=True)
        return df_zero.add(df, fill_value=0)

    @staticmethod
    def get_num_values(store, tensors, origin):
        dataset = store["row_meta"][["num_values"]]
        DataStatistics._to_numeric_row_id(dataset)
        dataset["col_id"] = dataset.apply(lambda x: np.arange(len(x[0])), axis=1)
        dataset = dataset.explode(["col_id", "num_values"]).set_index("col_id", append=True)
        dataset["num_values"] = dataset["num_values"].astype(int)
        return dataset

    @staticmethod
    def _to_numeric_row_id(dataset):
        table_ids = dataset.index.get_level_values(0)
        arange = np.arange(1, len(table_ids))
        arange[table_ids[1:] == table_ids[:-1]] = 0
        arange[np.where(arange == 0)[0][1:]] = arange[np.where(arange == 0)[0][1:] - 1]
        arange[np.where(arange == 0)[0][1:]] = arange[np.where(arange == 0)[0][1:] - 1]
        idx = np.arange(len(table_ids)) - np.hstack(([0], arange))
        dataset["row_id"] = idx
        dataset.index = dataset.index.droplevel("row_id")
        dataset.set_index("row_id", append=True, inplace=True)

    @staticmethod
    def get_origin(tensors, sample_ids=None):
        """Load the information information describing where each sample originated."""
        if sample_ids is None:
            origin = np.array(tensors["origin"]).astype(str)
        else:
            origin = np.array(tensors["origin"][sample_ids]).astype(str)
        origin = pd.DataFrame(origin, columns=["table_id", "table_name", "path"])
        origin["sample_id"] = origin.index if sample_ids is None else sample_ids
        origin.set_index("path", inplace=True)
        return origin

    @staticmethod
    def get_counts(df, group):
        """Update the counts."""
        counts = get_counts(df)
        for _, df in counts:
            col = next(iter(set(df.columns) - {"counts"}))
            DataStatistics.finalize_counts(col, group, df)
        return counts

    @staticmethod
    def finalize_counts(col, group, df):
        df["group"] = group
        df["col"] = col
        df.set_index(["col", "group", col], inplace=True)
        df.index.names = ["col", "group", "value"]

    def visualize_counts(self, file_path):
        """Generate plots."""
        pp = PdfPages(file_path.parent / (file_path.name.split(".")[0] + "-visualized.pdf"))
        plot_data = self.get_plot_data(file_path)

        for (plot_name, agg), group_data_pairs in sorted(plot_data.items()):
            full_df = reduce(lambda x, y: pd.concat((x, y)),
                             map(lambda x: pd.concat({x[0]: x[1]}, names=["group"]), group_data_pairs))
            all_groups = full_df.index.unique(level="group")
            suffixes = sorted(set("_".join(g.split("_")[1:]) for g in all_groups))
            for suffix in suffixes:
                groups = set([g for g in all_groups if g.endswith(suffix)])
                df = full_df.loc[sorted(groups)]
                values = df.index.unique(level="value")
                sorted_index_map = self.sorted(df)
                width = 1 / (len(groups) + 1)
                use_histogram = self.use_histogram(groups, df, sorted_index_map)
                for i, group in enumerate(sorted(groups)):
                    try:
                        offset = i * width - (width * len(groups) * .5) + 0.5 * width
                        data = df.loc[group]
                        sum_counts = data["counts"].sum()
                        plot_label = f"{group} ({sum_counts})"
                        if isinstance(data.index[0], str):
                            x_data = np.array([sorted_index_map[x] for x in data.index])
                        else:
                            x_data = data.index
                        if use_histogram:
                            plt.bar(x_data + offset, data["counts"] / sum_counts, width=width, label=plot_label)
                        else:
                            plt.scatter(x_data, data["counts"] / sum_counts, label=plot_label, s=10, rasterized=True)
                    except KeyError:
                        pass

                plt.legend()
                plt.title(f"Distribution of {plot_name} -- Grouped by {agg}")

                if plot_name in X_TICK_LABELS:
                    x_tick_labels = [X_TICK_LABELS[plot_name][v] for v in values]
                    plt.xticks(values, labels=x_tick_labels, rotation=10)
                plt.xlabel("value")
                plt.ylabel("p")
                pp.savefig(dpi=150)
                plt.clf()

        pp.close()

    def use_histogram(self, groups, df, sorted_index_map):
        for group in groups:
            try:
                data = df.loc[group]
                if isinstance(data.index[0], str):
                    x_data = np.array([sorted_index_map[x] for x in data.index])
                else:
                    x_data = data.index
                if max(x_data) > 15:
                    return False
            except KeyError:
                pass
        return True

    def get_plot_data(self, file_path):
        with h5py.File(file_path, "r") as file:
            group_names = list(file)
            plot_names = set([(plot_name, agg)
                              for group in group_names
                              for plot_name in file[group]["statistics"]
                              for agg in file[group]["statistics"][plot_name]])
            plot_data = dict()
            for plot_name, agg in plot_names:
                plot_data[plot_name, agg] = list()
                for group in group_names:
                    if plot_name not in file[group]["statistics"] or agg not in file[group]["statistics"][plot_name]:
                        continue
                    df = pd.read_hdf(file_path, key=f"{group}/statistics/{plot_name}/{agg}")
                    plot_data[plot_name, agg].append((group, df))
        return plot_data

    def export(self, counts, stats_path):
        """Export statistics to HDF5 and csv."""
        for key, dfs in counts.items():
            store_key = key
            if isinstance(key, tuple):
                store_key = "_".join(key)
            for (col, group), df in dfs.items():
                csv_dir = stats_path.parent / "stats" / stats_path.name.split(".")[0] / group / col
                csv_dir.mkdir(exist_ok=True, parents=True)
                store_df = df.loc[col, group]

                if isinstance(store_df.index[0], (int, float, np.integer, np.float)):
                    store_df.index = store_df.index.astype(int)
                else:
                    store_df.index = store_df.index.astype(str)
                store_df["counts"] = store_df["counts"].astype(int)
                store_df.to_csv(csv_dir / f"{store_key}.csv")
                store_df.to_hdf(stats_path, key=f"{group}/statistics/{col}/{store_key}", mode="a")

    def export_raw(self, tempdir, worker_id, i, df, group):
        tempfile = Path(tempdir) / f"{worker_id}-{i}.h5"
        col = set(df.columns) - {"sample_id", "table_id", "row_id", "col_id", "table_name"}
        assert len(col) == 1
        col = col.pop()
        if "table_name" in df:
            del df["table_name"]
        key = "/".join((col, group))
        df.to_hdf(tempfile, key=key, mode="a")
        logger.debug(f"Export statistics to {tempfile}")
        return tempfile


    def merge_non_aggregated(self, stats_path, tempfile):
        keys = list()
        with h5py.File(tempfile) as f:
            for col in f:
                for group in f[col]:
                    keys.append((col, group))
        for col, group in keys:
            df = pd.read_hdf(tempfile, "/".join((col, group)))
            key = f"{group}/detailed_statistics/{col}"
            min_itemsize = {c: 30 for c in df.columns if str(df[c].dtype) == "object"}
            for k, l in min_itemsize.items():
                df[k] = df[k].apply(_shorten(l))
            df.to_hdf(stats_path, key=key, mode="a", append=True, min_itemsize=min_itemsize)
        os.unlink(tempfile)

def _shorten(l):
    def f(s):
        encoded = [x.encode("utf-8") for x in s]
        cutoff = np.where(np.cumsum(list(map(len, encoded))) > l)[0]
        if len(cutoff):
            return s[:cutoff[0]]
        return s
    return f

class DataStatisticsStep(Step):
    """Compute statistics on training data."""

    depends_on = {DataCombineStep}

    def run(self, args, dataset):
        """Execute the step."""
        preprocessed_dataset = BaseLoader.get_final_preprocessed_data_path(args.dataset_dir, args.small_sample,
                                                                           dataset, args)
        data_statistics = DataStatistics(dataset, args.dataset_dir, args.small_sample, preprocessed_dataset)
        data_statistics.run(args.num_workers)


GatherFunc = namedtuple("GatherFunc", ["identifier", "callable", "Description"])
GATHER_FUNCS = [
    # Table Level
    GatherFunc("#windows", None, "In how many windows is each table-text pair divided?"),

    # Sample level
    GatherFunc("column ids", DataStatistics.get_header_column_ids,
               "How often is which column used?"),
    GatherFunc("row ids", DataStatistics.get_row_ids,
               "How often is which entity used as a row?"),
    GatherFunc("queried", DataStatistics.get_queried_attrs,
               "How often is which attribute queried?"),
    GatherFunc("table types", DataStatistics.get_table_types,
               "How often is which Wikidata-type used as table type?"),
    GatherFunc("database operator", DataStatistics.get_tbl_stats("db_operator"),
               "How often is which DB operator used?"),
    GatherFunc("#rows", DataStatistics.get_example_tbl_stats("table_size", 0, "num_rows"),
               "How many rows does each sample have?"),
    GatherFunc("#columns", DataStatistics.get_example_tbl_stats("table_size", 1, "num_cols"),
               "How many columns does each sample have?"),

    # Row level
    GatherFunc("#masked cells", DataStatistics.get_example_row_stats("num_cells_masked"),
               "Num masked cells "),

# Row level:
# Number of overlapped cells?  # TODO
# Number of dependent queries?
# Number of non-dependent queries?
# Number of header queries?
# Number of context tokens masked?
# Number of topics?

    # Query Level
    GatherFunc("#num answers", DataStatistics.get_num_answers,
              "How many answers does per masked cell"),

    # Cell level
    GatherFunc("#tokens per cell", DataStatistics.get_num_cell_tokens,
               "How many tokens does each cell have."),
    GatherFunc("#values per cell", DataStatistics.get_num_values,
               "How many values does each cell have."),
]
