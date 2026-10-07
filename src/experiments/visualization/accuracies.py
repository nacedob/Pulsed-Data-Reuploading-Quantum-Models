"""Visualization utilities for experiment results.

Provides plotting and aggregation for layer-depth and noise experiments,
together with three-dimensional dataset visualizations.

When run as a module, experiment results are loaded, merged copies are
written to the results directory, and figures are generated for each dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import get_root_path
from src.experiments.config_exp import get_dataset
from .utils import STYLES

FONTSIZE = 34
METRIC = "accuracy"
DEFAULT_MODELS = ["mixed", "gate"]
DEFAULT_LAYERS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
DATASET_LIST = ["digits_08", "helix", "iris", "corners3d"]


def _configure_matplotlib() -> None:
    """Apply the common LaTeX-based style used by experiment figures."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


def dataset_mapping(dataset: str) -> str:
    """Return the human-readable name used for a dataset in figure titles."""
    if "digits" in dataset:
        parts = dataset.split("_")
        if len(parts) != 2 or len(parts[1]) != 2:
            raise ValueError(f"Unexpected digits dataset format: {dataset}")
        return f"Digits {parts[1][0]}-{parts[1][1]}"

    return {
        "fashion_mnist": "Fashion-MNIST",
        "corners3d": "Corners",
        "iris": "Iris",
        "helix": "Helix",
    }.get(dataset, dataset)


def get_column(metric: str, data_type: str) -> str:
    """Return the dataframe column for a metric and train/test split."""
    if data_type not in {"train", "test"}:
        raise ValueError("data_type must be 'train' or 'test'")

    metric = metric.lower()
    if metric == "loss":
        return f"{data_type}_loss"
    if metric in {"acc", "accuracy"}:
        return f"acc_{data_type}"

    raise ValueError("metric must be 'loss', 'acc', or 'accuracy'")


def get_metric_columns(metric: str) -> tuple[str, str]:
    """Return the train and test dataframe columns for a metric."""
    return get_column(metric, "train"), get_column(metric, "test")


def build_block(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Convert a historical wide-format result table into the common format."""
    return pd.DataFrame(
        {
            "n_qubits": df["n_qubits"],
            "n_layers": df["n_layers"],
            "seed": df["seed"],
            "dataset": df["dataset"],
            "model": model,
            "p": df["p"] if "p" in df.columns else None,
            "train_loss": df[f"{model}_train_loss"],
            "test_loss": np.nan,
            "acc_train": df[f"{model}_acc_train"],
            "acc_test": df[f"{model}_acc_test"],
            "lr": df[f"{model}_lr"],
        }
    )


def format_previous_results(previous_path: Path) -> pd.DataFrame:
    """Load a historical CSV containing gate and mixed model results."""
    df = pd.read_csv(previous_path)
    return pd.concat(
        [build_block(df, "gate"), build_block(df, "mixed")],
        ignore_index=True,
    )


# Keep the historical function name available for callers that use it.
format_previous_noise = format_previous_results


def merge_noise(results_folder: str | Path | None = None) -> pd.DataFrame:
    """Load historical and current noise results into one dataframe.

    Current results are read from ``NOISE_EXPERIMENT``. Each device directory
    is expected to contain a ``depolarizing_1q_<p>`` suffix, and results may be
    either complete CSV files or CSV files inside a ``partial`` directory.
    """
    root = (
        Path(results_folder)
        if results_folder is not None
        else Path(get_root_path()) / "data" / "results"
    )
    previous_folder = root / "previous"
    noise_folder = root / "NOISE_EXPERIMENT"

    frames = [
        format_previous_results(previous_folder / "all_noise_digits_08.csv")
    ]

    if noise_folder.exists():
        for device in noise_folder.iterdir():
            if not device.is_dir() or "depolarizing_1q_" not in device.name:
                continue

            p = float(device.name.split("depolarizing_1q_", 1)[1])

            for dataset in device.iterdir():
                if not dataset.is_dir():
                    continue

                for result in dataset.iterdir():
                    if result.is_file() and result.name.startswith("results") and result.suffix == ".csv":
                        df = pd.read_csv(result)
                        df["p"] = p
                        frames.append(df)
                    elif result.is_dir() and result.name == "partial":
                        for partial_result in result.iterdir():
                            if partial_result.is_file() and partial_result.suffix == ".csv":
                                df = pd.read_csv(partial_result)
                                df["p"] = p
                                frames.append(df)

    return pd.concat(frames, ignore_index=True)


def merge_layers(results_folder: str | Path | None = None) -> pd.DataFrame:
    """Load historical and current layer-depth results into one dataframe."""
    root = (
        Path(results_folder)
        if results_folder is not None
        else Path(get_root_path()) / "data" / "results"
    )
    previous_folder = root / "previous"
    layer_folder = root / "LAYERS_EXPERIMENT" / "default"

    frames = [
        format_previous_results(previous_folder / "all_layers_digits_08.csv")
    ]

    if layer_folder.exists():
        for dataset in layer_folder.iterdir():
            if not dataset.is_dir():
                continue

            for result in dataset.iterdir():
                if result.is_file() and result.name.startswith("results") and result.suffix == ".csv":
                    frames.append(pd.read_csv(result))
                elif result.is_dir() and result.name == "partial":
                    for partial_result in result.iterdir():
                        if partial_result.is_file() and partial_result.suffix == ".csv":
                            frames.append(pd.read_csv(partial_result))

    return pd.concat(frames, ignore_index=True)


def aggregate(
    df: pd.DataFrame,
    metric: str,
    x_col: str,
) -> pd.DataFrame:
    """Calculate train/test mean and standard deviation by x value and model."""
    train_col, test_col = get_metric_columns(metric)
    return (
        df.groupby([x_col, "model"])
        .agg(
            {
                train_col: ["mean", "std"],
                test_col: ["mean", "std"],
            }
        )
        .reset_index()
    )


def build_experiment_data(
    df_layers: pd.DataFrame,
    df_noise: pd.DataFrame,
    models: list[str],
    metric: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Prepare layer and noise statistics in a common plotting format."""
    data: dict[tuple[str, str, str], dict[str, Any]] = {}

    for experiment_name, df, x_column in (
        ("layers", df_layers, "n_layers"),
        ("noise", df_noise, "p"),
    ):
        aggregated = aggregate(df, metric, x_column)

        for model in models:
            if model not in STYLES:
                raise ValueError(
                    f"Unknown model '{model}'. Available models: {list(STYLES)}"
                )

            subset = aggregated[aggregated["model"] == model]
            for split in ("train", "test"):
                metric_column = get_column(metric, split)
                data[(experiment_name, model, split)] = {
                    "x": subset[x_column],
                    "mean": subset[(metric_column, "mean")],
                    "std": subset[(metric_column, "std")],
                    "style": STYLES[model],
                }

    return data


def _configure_experiment_axes(
    axes: np.ndarray,
    experiment_name: str,
    metric: str,
) -> None:
    """Configure labels, limits, ticks, and grid lines for experiment axes."""
    xlabel = (
        "Number of Layers"
        if experiment_name == "layers"
        else "Depolarizing probability (p)"
    )
    ylabel = "Loss" if metric == "loss" else "Accuracy"

    for axis in axes:
        axis.set_xlabel(xlabel, fontsize=FONTSIZE - 2)
        axis.grid(True, ls=":", lw=0.6)
        axis.tick_params(axis="both", labelsize=FONTSIZE - 4)

    axes[0].set_ylabel(ylabel, fontsize=FONTSIZE - 2)

    if metric != "loss":
        for axis in axes:
            axis.set_ylim(0.4, 1.0)
            axis.set_yticks([0.4, 0.6, 0.8, 1.0])

    if experiment_name == "layers":
        for axis in axes:
            axis.set_xlim(0, 41)
            axis.set_xticks([0, 10, 20, 30, 40])
    else:
        for axis in axes:
            axis.set_xlim(-0.005, 0.305)
            axis.set_xticks([0.0, 0.1, 0.2, 0.3])


def plot_experiment(
    data: dict[tuple[str, str, str], dict[str, Any]],
    models: list[str],
    metric: str,
    save_folder: str | Path,
    dataset: str,
    exp_name: str | None = None,
) -> None:
    """Create train/test figures for layer and/or noise experiments."""
    _configure_matplotlib()
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    experiment_names = [exp_name] if exp_name is not None else ["layers", "noise"]

    for experiment_name in experiment_names:
        if experiment_name not in {"layers", "noise"}:
            raise ValueError("exp_name must be 'layers', 'noise', or None")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for axis, split in zip(axes, ("train", "test")):
            for model in models:
                key = (experiment_name, model, split)
                if key not in data:
                    continue

                experiment = data[key]
                style = experiment["style"]
                x = np.asarray(experiment["x"], dtype=float)
                mean = np.asarray(experiment["mean"], dtype=float)
                std = np.asarray(experiment["std"], dtype=float)

                order = np.argsort(x)
                x, mean, std = x[order], mean[order], std[order]

                axis.plot(
                    x,
                    mean,
                    label=f"{style['name']} - {split.capitalize()}",
                    linestyle="-" if split == "train" else "--",
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=3,
                    markersize=10,
                )
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=0.1,
                    color=style["color"],
                )

            axis.set_title(
                "Training" if split == "train" else "Testing",
                fontsize=FONTSIZE,
            )

        _configure_experiment_axes(axes, experiment_name, metric)

        fig.suptitle(
            rf"\textbf{{{dataset_mapping(dataset)}}}",
            fontsize=FONTSIZE + 2,
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

        save_path = save_folder / experiment_name / f"{dataset}_{metric}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=500)
        plt.close(fig)
        print(f"Saved {save_path}")


def main(
    dataset: str = "digits_08",
    metric: str = METRIC,
    df_layers: pd.DataFrame | None = None,
    df_noise: pd.DataFrame | None = None,
    models: list[str] | None = None,
    layers: list[int] | None = None,
    save_folder: str | Path | None = None,
    results_folder: str | Path | None = None,
) -> None:
    """Generate layer and noise figures for one dataset.

    When dataframes are not supplied, results are loaded using the project's
    standard historical and current experiment directories.
    """
    models = models or DEFAULT_MODELS
    layers = layers or DEFAULT_LAYERS

    if df_layers is None:
        df_layers = merge_layers(results_folder)
    if df_noise is None:
        df_noise = merge_noise(results_folder)

    df_layers = df_layers[df_layers["dataset"] == dataset].copy()
    df_noise = df_noise[df_noise["dataset"] == dataset].copy()
    df_layers = df_layers[df_layers["n_layers"].isin(layers)]

    experiment_data = build_experiment_data(
        df_layers=df_layers,
        df_noise=df_noise,
        models=models,
        metric=metric,
    )

    plot_experiment(
        data=experiment_data,
        models=models,
        metric=metric,
        save_folder=(
            save_folder
            or Path(get_root_path()) / "data" / "results" / "figures"
        ),
        dataset=dataset,
    )


def plot_all_results(
    metric: str = METRIC,
    save_folder: str | Path | None = None,
    results_folder: str | Path | None = None,
    layers_limit: int = 40,
) -> None:
    """Merge all experiment results and generate figures for every dataset."""
    noise_df = merge_noise(results_folder)
    layers_df = merge_layers(results_folder)
    layers_df = layers_df[layers_df["n_layers"] <= layers_limit].copy()

    # Filter by dataset
    noise_df = noise_df[noise_df["dataset"].isin(DATASET_LIST)]
    layers_df = layers_df[layers_df["dataset"].isin(DATASET_LIST)]

    root = (
        Path(results_folder)
        if results_folder is not None
        else Path(get_root_path()) / "data" / "results"
    )
    merged_folder = root / "merged"
    merged_folder.mkdir(parents=True, exist_ok=True)
    noise_df.to_csv(merged_folder / "noise.csv", index=False)
    layers_df.to_csv(merged_folder / "layers.csv", index=False)

    datasets = sorted(
        set(noise_df["dataset"].unique()) | set(layers_df["dataset"].unique())
    )

    output_folder = save_folder or root / "figures"
    for dataset in datasets:
        main(
            dataset=dataset,
            metric=metric,
            df_layers=layers_df,
            df_noise=noise_df,
            save_folder=output_folder,
            results_folder=results_folder,
        )


def plot_dataset(
    dataset: str,
    save_folder: str | Path = "data/results/figures/datasets",
    elev: float = 20,
    azim: float = 45,
    s: float = 20,
) -> None:
    """Plot a three-dimensional training dataset and save it as a PNG."""
    X_train, y_train, _, _ = get_dataset(
        dataset=dataset,
        n_train=3000,
        n_test=3,
        points_dimension=3,
        seed=0,
        interface="jax",
    )

    X = np.asarray(X_train)
    y = np.asarray(y_train)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be of shape (N, 3)")

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for class_value, model_style in ((0, STYLES["mixed"]), (1, STYLES["gate"])):
        mask = y == class_value
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            X[mask, 2],
            c=model_style["color"],
            s=s,
            label=f"Class {class_value}",
            alpha=0.6,
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    save_path = save_folder / f"{dataset}.png"
    fig.savefig(save_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved {dataset} plot to {save_path}")


def plot_legend(
    save_folder: str | Path = "data/results/figures/datasets",
) -> None:
    """Save a standalone class legend for the 3D dataset figures."""
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(3, 1.5))
    ax = fig.add_subplot(111)
    ax.scatter([], [], c=STYLES["mixed"]["color"], label="Class 0")
    ax.scatter([], [], c=STYLES["gate"]["color"], label="Class 1")
    ax.axis("off")
    ax.legend(loc="center", fontsize=14, ncol=2, frameon=True)

    fig.tight_layout()
    save_path = save_folder / "legend.png"
    fig.savefig(save_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved legend plot to {save_path}")


if __name__ == "__main__":
    plot_all_results()
