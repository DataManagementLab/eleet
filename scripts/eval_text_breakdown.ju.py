# %%
from typing import Optional, Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
### Load stats

# %%
path_to_stats = "/home/murban/multimodal-database/predictions/test/stats_eleet-t2t-llama-gpt-3.5-turbo-0125-gpt-3.5-turbo-0125:4-gpt-4-0613-llama-ft-gpt-ft-rotowire-trex-aviation-corona-64-test-limit-4294967296.csv"

stats = pd.read_csv(path_to_stats)

stats["recall"] = stats["tp"] / (stats["tp"] + stats["fn"])  # type: ignore
stats["precision"] = stats["tp"] / (stats["tp"] + stats["fp"])  # type: ignore
stats["f1"] = 2 * stats["precision"] * stats["recall"] / (stats["precision"] + stats["recall"])  # type: ignore

filter_queries = stats["query"].unique()  # type: ignore
filters = {
    "selection": lambda x: "σ" in x,
    "aggregate": lambda x: "G_" in x,
}
filter_queries = set(filter(lambda x: not any(f(x) for f in filters.values()), filter_queries))

print(filter_queries)

stats

# %% [markdown]

### Plot

# %%

def plot_stats(stats: pd.DataFrame, filter_queries: Optional[set], smoothing: int = 10,
               text_length_column: str = "num_tokens"):
    methods = {"ELEET": "ELEET",
               "Text-To-Table": "Text-To-Table",
               "gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
               "LLaMA": "LLaMA-2 7B (ic)",
               "LLaMA-FT": "LLaMA-2 7B (ft)"}
    palette = sns.color_palette()
    color_matching = {
        "ELEET": palette[0],  # blue
        "Text-To-Table": palette[2],  # green
        "LLaMA-2 7B (ic)": palette[1],  # orange
        "LLaMA-2 7B (ft)": palette[3],  # red
        "gpt-3.5-turbo-0125": palette[4],  #purple
    }
    color_palette = sns.color_palette([color_matching[m] for m in methods.values()])
    this_stats = stats[stats["method"].isin(methods)]  # type: ignore 
    if filter_queries is not None:
        this_stats = this_stats[this_stats["query"].isin(filter_queries)]  # type: ignore

    pos_map = this_stats.groupby(["dataset", "idx"])[[text_length_column]].mean().reset_index(["idx"])  # type: ignore
    pos_map = pos_map.sort_values([text_length_column, "idx"])  # type: ignore
    max_pos = pos_map.groupby("dataset").size() # type: ignore
    pos_map = pos_map.groupby("dataset").apply(  # type: ignore
            lambda x: pd.Series({int(row["idx"]): i for i, (_, row) in enumerate(x.iterrows())})
    )  # type: ignore
    pos_map.index.names = ["dataset", "idx"]
    pos_map.name = "pos"
    max_pos.name = "max_pos"
    this_stats = this_stats.join(pos_map, on=["dataset", "idx"]).join(max_pos, on="dataset")  # type: ignore
    this_stats["bucket"] = ((this_stats["max_pos"] - this_stats["pos"]) // smoothing).astype(int)  # type: ignore
    bucket_boundaries = this_stats.groupby(["dataset", "bucket"])[text_length_column].agg(["min", "max"])
    bucket_boundaries["bucket_center"] = (bucket_boundaries["min"] + bucket_boundaries["max"]) / 2  # type: ignore
    bucket_boundaries.columns = ["bucket_min", "bucket_max", "bucket_center"]  # type: ignore
    bucket_boundaries = bucket_boundaries.astype(int, errors="ignore")  # type: ignore
    this_stats = this_stats.join(bucket_boundaries, on=["dataset", "bucket"])  # type: ignore

    # rename methods
    this_stats["method"] = this_stats["method"].map(methods)  # type: ignore

    fig = plt.figure(figsize=(8, 2.5))
    gs = fig.add_gridspec(nrows=1, ncols=2, hspace=0.05, wspace=0.05)
    axes = gs.subplots(sharey=True)
    labels = list()
    handles = list()

    for i, dataset in enumerate(("rotowire", "trex")):  # , "aviation", "corona")):
        ax = axes[i]
        sns.lineplot(data=this_stats[this_stats["dataset"] == dataset], x=text_length_column, y="f1", hue="method", ax=ax,  # type: ignore
                     alpha=0.2, hue_order=list(methods.values())[::-1], palette=color_palette[::-1], errorbar=None)
        sns.lineplot(data=this_stats[this_stats["dataset"] == dataset], x="bucket_max", y="f1", hue="method", ax=ax,  # type: ignore
                     hue_order=list(methods.values())[::-1], palette=color_palette[::-1], errorbar=None)

        ax.plot([512, 512], [0, 1], color="black", linestyle="--", alpha=0.5, label="Context Length of ELEET")

        handles_ax, labels_ax = ax.get_legend_handles_labels()
        handles.extend(handles_ax)
        labels.extend(labels_ax)
        ax.get_legend().remove()


        # set label of x and y axis
        ax.set_title(dataset)
        ax.set_xlabel(" ".join([x.capitalize() for x in text_length_column.split("_")]))
        ax.set_ylabel("F1 Score" if i == 0 else "")

    hl_dict = {l.split("_")[0]: h for l, h in zip(labels, handles)}
    methods["Context Length of ELEET"] = "Context Length of ELEET"
    print(hl_dict)

    fig.legend([hl_dict[l] for l in methods.values()],
               methods.values(), bbox_to_anchor=(0.91 , 0.1), ncol=1, loc='lower left', borderaxespad=0., frameon=False)

    plt.savefig(f"text-breakdown.pdf", bbox_inches="tight")
    plt.show()
    return False


plot_stats(stats, filter_queries)  # type: ignore

# %%

# for dataset in stats["dataset"].unique():  # type: ignore
#     this_queries = stats[stats["dataset"] == dataset]["query"].unique()  # type: ignore
#     for query in filter_queries & set(this_queries):
#         plot_stats(stats, dataset, query)  # type: ignore

# %%

