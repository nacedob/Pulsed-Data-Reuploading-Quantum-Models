import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import get_root_path

# --- Config ---
ROOT = Path(get_root_path())
EXPERIMENT = 'LAYERS'
CSV_PATH = ROOT / f"data/results/{EXPERIMENT}_EXPERIMENT/default"
OUTPUT_FOLDER = ROOT / f"data/results/figures/{EXPERIMENT}_EXPERIMENT"
METRIC_TRAIN = "acc_train"
METRIC_TEST = "acc_test"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
# --- Load ---


def load_data() -> pd.DataFrame:
    """
    Recursively loads all CSV files under csv_root.
    Adds a 'dataset' column based on the first-level directory.
    Returns a single concatenated DataFrame.
    """
    csv_root = Path(CSV_PATH)

    dfs = []

    for dataset_dir in csv_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Recursively find all CSVs inside this dataset folder
        for csv_file in dataset_dir.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df["dataset"] = dataset_name
                dfs.append(df)
            except Exception as e:
                print(f"Skipping {csv_file}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

def get_metric_columns(metric: str):
    if metric == "loss":
        return "train_loss", "test_loss"
    elif metric == "acc":
        return "acc_train", "acc_test"
    else:
        raise ValueError("metric must be 'loss' or 'acc'")
def aggregate(df, metric: str):
    train_col, test_col = get_metric_columns(metric)

    train = (
        df.groupby(["dataset", "n_layers", "model"])[train_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    test = (
        df.groupby(["dataset", "n_layers", "model"])[test_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    return train, test

def plot(df: pd.DataFrame, metric: str = "acc", save_path: str = "./plot.png") -> None:
    train_col, test_col = get_metric_columns(metric)

    models = df["model"].unique()
    colors = {"gate": "#6a3d9a", "mixed": "#e7298a"}

    train_stats = (
        df.groupby(["n_layers", "model"])[train_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    test_stats = (
        df.groupby(["n_layers", "model"])[test_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, stats, title in zip(
        axes,
        [train_stats, test_stats],
        ["Train", "Test"]
    ):

        for model in models:
            sub = stats[stats["model"] == model].sort_values("n_layers")

            x = sub["n_layers"].values
            y = sub["mean"].values
            yerr = sub["std"].values

            ax.plot(
                x, y,
                marker="o",
                linestyle="--" if model == "mixed" else "-",
                color=colors.get(model),
                label=model
            )

            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=colors.get(model),
                alpha=0.2
            )

        ax.set_title(title)
        ax.set_xlabel("Number of layers")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models))

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f'Saved plot to {save_path}')

if __name__ == "__main__":
    df = load_data()
    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset]
        plot(df_dataset, metric="loss", save_path=OUTPUT_FOLDER / f"{dataset}.png")
