import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic

from config import get_root_path
from src.experiments.visualization.utils import styles

FONTSIZE = 24

plt.rcParams.update({
    "text.usetex": True,              # activa LaTeX real
    "font.family": "serif",           # fuente serif tipo LaTeX
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# =============================================================================
# Utility functions
# =============================================================================
def get_column(metric: str, data_type: str) -> str:
    if metric == "loss":
        return "train_loss" if data_type == "train" else "test_loss"

    return "acc_train" if data_type == "train" else "acc_test"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(get_root_path())
    folder = root / "data/results/previous"

    df_layers = pd.read_csv(folder / "all_layers_digits_08_filled.csv")
    df_noise = pd.read_csv(folder / "all_noise_digits_08.csv")

    return df_layers, df_noise


# =============================================================================
# Aggregation
# =============================================================================
def aggregate(df: pd.DataFrame, metric: str, x_col: str) -> pd.DataFrame:
    return (
        df.groupby([x_col, "model"])
        .agg({
            get_column(metric, "train"): ["mean", "std"],
            get_column(metric, "test"): ["mean", "std"],
        })
        .reset_index()
    )


def aggregate_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate noise experiment statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw noise experiment dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe containing mean and standard deviation
        for each model/metric combination.
    """
    ic(df.columns)
    df_long = df.melt(
        id_vars=["p", "dataset"],
        value_vars=[
            "mixed_train_loss",
            "mixed_test_loss",
            "gate_train_loss",
            "gate_test_loss",
        ],
        var_name="tmp",
        value_name="value",
    )

    df_long[["model", "metric"]] = df_long["tmp"].str.extract(
        r"(\w+)_(.*)"
    )

    df_long = df_long.drop(columns="tmp")

    return (
        df_long.groupby(["p", "model", "metric"])
        .agg(mean=("value", "mean"), std=("value", "std"))
        .reset_index()
    )


# =============================================================================
# Data preparation
# =============================================================================
def build_experiment_data(
    df_layers: pd.DataFrame,
    df_noise: pd.DataFrame,
    models: list[str],
    metric: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """
    Build a unified plotting dictionary for experiments.

    Parameters
    ----------
    df_layers : pd.DataFrame
        Aggregated layers dataframe.

    df_noise : pd.DataFrame
        Aggregated noise dataframe.

    models : list[str]
        Models to include.

    metric : str
        Metric to visualize ("loss" or "acc").

    Returns
    -------
    dict
        Nested dictionary containing plotting information.
    """
    data: dict[tuple[str, str, str], dict[str, Any]] = {}

    experiments = {
        "layers": {
            "df": df_layers,
            "x": "n_layers",
        },
        "noise": {
            "df": df_noise,
            "x": "p",
        },
    }

    for exp_name, exp_config in experiments.items():
        df = exp_config["df"]
        x_column = exp_config["x"]

        # Aggregation
        df = aggregate(df, metric, x_column)

        for model in models:
            subset = df[df["model"] == model]
            for split in ("train", "test"):
                metric_column = get_column(metric, split)

                data[(exp_name, model, split)] = {
                    "x": subset[x_column],
                    "mean": subset[(metric_column, "mean")],
                    "std": subset[(metric_column, "std")],
                    "style": styles[model],
                }

    return data


# =============================================================================
# Plotting
# =============================================================================

def dataset_mapping(dataset: str) -> str:
    if 'digits' in dataset:
        comparison = dataset.split('_')[1]
        assert len(comparison) == 2, f"Unexpected digits dataset format: {dataset}"
        return f"Digits {comparison[0]}-{comparison[1]}"
    mapping = {
        "fashion_mnist": "Fashion-MNIST",
        "corners3d": "Corners 3D",
        "iris": "Iris",
        "helix": "Helix",
    }
    return mapping.get(dataset, dataset)

def plot_experiment(
    data: dict[tuple[str, str, str], dict[str, Any]],
    models: list[str],
    metric: str,
    save_folder: str,
    dataset: str,
    exp_name: str | None = None,
) -> None:
    """
    Plot train/test experiment results.

    Parameters
    ----------
    exp_name : str
        Experiment name ("layers" or "noise").

    data : dict
        Plot-ready experiment data.

    models : list[str]
        Models to display.

    metric : str
        Metric being plotted.

    save_folder : str
        Output directory for the generated figure.
    """
    exp_list = [exp_name] if exp_name is not None else ["layers", "noise"]
    os.makedirs(save_folder, exist_ok=True)

    for exp_name in exp_list:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        for axis, split in zip(axes, ["train", "test"]):
            for model in models:
                key = (exp_name, model, split)

                if key not in data:
                    continue

                experiment = data[key]
                style = experiment["style"]

                axis.plot(
                    experiment["x"],
                    experiment["mean"],
                    label=f"{style['name']} - {split.capitalize()}",
                    linestyle="-" if split == "train" else "--",
                    color=style["color"],
                    marker=style["marker"],
                )

                axis.fill_between(
                    experiment["x"],
                    experiment["mean"] - experiment["std"],
                    experiment["mean"] + experiment["std"],
                    alpha=0.1,
                    color=style["color"],
                )

            split = "Training" if split == "train" else "Testing"
            axis.set_title(split.capitalize(), fontsize=FONTSIZE)
            axis.grid(True, ls=":", lw=0.6)
            axis.tick_params(axis="both", labelsize=FONTSIZE - 4)

        xlabel = (
            "Number of Layers"
            if exp_name == "layers"
            else "Depolarizing probability (p)"
        )

        ylabel = "Loss" if metric == "loss" else "Accuracy"

        axes[0].set_xlabel(xlabel, fontsize=FONTSIZE - 2)
        axes[1].set_xlabel(xlabel, fontsize=FONTSIZE - 2)
        axes[0].set_ylabel(ylabel, fontsize=FONTSIZE - 2)
        
        if metric != "loss":
            axes[0].set_ylim(0.5, 1.0)
            axes[1].set_yticks([0.5, 0.75, 1.0])
            axes[1].set_ylim(0.5, 1.0)
            axes[1].set_yticks([0.5, 0.75, 1.0])
        if exp_name == "layers":
            axes[0].set_xlim(0, 50)
            axes[1].set_xlim(0, 50)
            axes[0].set_xticks([0, 10, 20, 30, 40, 50])
            axes[1].set_xticks([0, 10, 20, 30, 40, 50])
        else:
            axes[0].set_xlim(0.0, 0.3)
            axes[1].set_xlim(0.0, 0.3)
            axes[0].set_xticks([0.0, 0.1, 0.2, 0.3])
            axes[1].set_xticks([0.0, 0.1, 0.2, 0.3])
            

        fig.suptitle(
            f"Dataset: {dataset_mapping(dataset)}",
            fontsize=FONTSIZE,
        )

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=FONTSIZE - 4,
        )

        fig.tight_layout(rect=[0, 0.1, 1, 1])

        save_path = os.path.join(
            save_folder,
            f"{exp_name}_{metric}.png",
        )

        fig.savefig(save_path, dpi=500)

        print(f"Saved {save_path}")

        plt.close(fig)


# =============================================================================
# Main execution
# =============================================================================
def main(
    dataset: str = "digits_08",
    metric: str = "loss",
    df_layers: pd.DataFrame | None = None,
    df_noise: pd.DataFrame | None = None,
    models: list[str] | None = None,
    layers: list[int] | None = None,
    save_folder: str | None = None,
) -> None:
    """
    Run experiment aggregation and visualization pipeline.

    Parameters
    ----------
    dataset : str, optional
        Dataset identifier.

    metric : str, optional
        Metric to visualize ("loss" or "acc").

    df_layers : pd.DataFrame | None, optional
        Preloaded layers dataframe.

    df_noise : pd.DataFrame | None, optional
        Preloaded noise dataframe.

    models : list[str] | None, optional
        Models to include in the plots.

    layers : list[int] | None, optional
        Layer values to keep.

    save_folder : str | None, optional
        Output directory for figures.
    """
    models = models or ["mixed", "gate"]
    layers = layers or [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    if df_layers is None or df_noise is None:
        df_layers, df_noise = load()

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    df_layers = df_layers[df_layers["dataset"] == dataset]
    df_noise = df_noise[df_noise["dataset"] == dataset]

    df_layers = df_layers[df_layers["n_layers"].isin(layers)]

    # -------------------------------------------------------------------------
    # Build plotting data
    # -------------------------------------------------------------------------
    experiment_data = build_experiment_data(
        df_layers=df_layers,
        df_noise=df_noise,
        models=models,
        metric=metric,
    )

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    plot_experiment(
        data=experiment_data,
        models=models,
        metric=metric,
        save_folder=save_folder or ".",
        dataset=dataset,
    )


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()
