"""
Plotting utilities for concept residual model evaluation results.

This module provides functions for generating plots from Ray Tune experiment
results, including intervention curves, disentanglement metrics, concept
prediction accuracy, and attribution analysis.
"""

from __future__ import annotations

import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from pathlib import Path
from ray import tune
from ray.train import Result
from ray.tune import ResultGrid
from tqdm import tqdm
from typing import Callable

from evaluate import evaluate
from lightning_ray import group_results


### Helper Functions


def format_plot_title(plot_key: str | tuple[str, ...] | list[str]) -> str:
    """
    Get a nicely-formatted title for the given dataset.

    Parameters
    ----------
    plot_key : str or tuple[str]
        Plot key to format
    """
    if isinstance(plot_key, (list, tuple)):
        if len(plot_key) > 1:
            return ", ".join([format_plot_title(key) for key in plot_key])
        else:
            plot_key = plot_key[0]

    if isinstance(plot_key, str):
        plot_key = plot_key.replace("_", " ").title()
        plot_key = plot_key.replace("Mnist", "MNIST")
        plot_key = plot_key.replace("Cifar", "CIFAR")
        plot_key = plot_key.replace("Cub", "CUB")
        plot_key = plot_key.replace("Oai", "OAI")
        plot_key = plot_key.replace("Mi Residual", "Mutual Info Residual")
        plot_key = plot_key.replace("Iter Norm", "IterNorm Residual")

    return str(plot_key)


def get_save_path(
    plot_key: tuple,
    prefix: str | None = None,
    suffix: str | None = None,
    save_dir: Path | str = "./plots",
) -> Path:
    """
    Get the save path for the given plot.
    """
    items = [str(key).replace(".", "_") for key in plot_key]
    if prefix:
        items.insert(0, prefix)
    if suffix:
        items.append(suffix)

    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir / "_".join(items)


def plot_curves(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str],
    title: str,
    x_label: str,
    y_label: str,
    get_x: Callable[[ResultGrid], np.ndarray],
    get_y: Callable[[Result], np.ndarray],
    eval_mode: str,
    save_dir: Path | str,
    save_name: str,
    prefix: str | None = None,
    show: bool = True,
):
    """
    Create a plot with curve(s) for the specified results.
    """
    plt.clf()
    save_path = get_save_path(
        plot_key, prefix=prefix, suffix=save_name, save_dir=save_dir
    )

    data, columns = [], []
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    for key, results in plot_results.items():
        results = group_results(results, groupby="eval_mode")
        if eval_mode not in results:
            tqdm.write(f"No {eval_mode} results found for: {plot_key} {groupby} {key}")
            continue

        x = get_x(results[eval_mode])
        y = np.stack([get_y(result) for result in results[eval_mode]]).mean(axis=0)
        y_std = np.stack([get_y(result) for result in results[eval_mode]]).std(axis=0)
        plt.plot(x, y, label=key)
        data.extend([y, y_std])
        columns.extend([f"{key} {y_label}", f"{key} {y_label} Std."])

    # Create CSV file
    x = np.linspace(0, 1, len(data[0]))
    data = np.stack([x, *data], axis=1)
    columns.insert(0, x_label)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Create figure
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_scatter(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str],
    title: str,
    x_label: str,
    y_label: str,
    x_eval_mode: str,
    y_eval_mode: str,
    get_x: Callable[[ResultGrid], float | np.ndarray],
    get_y: Callable[[ResultGrid], float | np.ndarray],
    save_dir: Path | str,
    save_name: str,
    prefix: str | None = None,
    show_regression_line: bool = False,
    show: bool = True,
):
    """
    Create a scatter plot for the specified results.
    """
    plt.clf()
    save_path = get_save_path(
        plot_key, prefix=prefix, suffix=save_name, save_dir=save_dir
    )

    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    x_values, y_values, index = [], [], []
    for key in plot_results.keys():
        results = group_results(plot_results[key], groupby="eval_mode")
        if x_eval_mode not in results or np.isnan(get_x(results[x_eval_mode])).any():
            tqdm.write(
                f"No {x_eval_mode} results found for: {plot_key} {groupby} {key}"
            )
            continue
        if y_eval_mode not in results or np.isnan(get_y(results[y_eval_mode])).any():
            tqdm.write(
                f"No {y_eval_mode} results found for: {plot_key} {groupby} {key}"
            )
            continue

        x = get_x(results[x_eval_mode])
        y = get_y(results[y_eval_mode])
        plt.scatter(x, y, label=key)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_values.extend(x.flatten())
            y_values.extend(y.flatten())
            index += [key] * len(x)
        else:
            x_values.append(x)
            y_values.append(y)
            index.append(key)

    if len(x_values) == 0 or len(y_values) == 0:
        tqdm.write(f"No {save_name} results found for: {plot_key}")
        return

    # Create CSV file
    data = np.stack([x_values, y_values], axis=1)
    df = pd.DataFrame(data, index=index, columns=[x_label, y_label])
    df.to_csv(save_path.with_suffix(".csv"), index=True)

    # Add linear regression line
    if show_regression_line:
        from scipy.stats import linregress

        x, y = np.array(x_values), np.array(y_values)
        m, b, r, _, _ = linregress(x, y)
        plt.plot(x, m * x + b, color="black")
        plt.text(
            x=0.5,
            y=0.5,
            s=f"$r^2$ = {r**2:.3f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.75),
        )

    # Create figure
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path.with_suffix(".pdf"))
    if show:
        plt.show()


### Plotting


def plot_negative_interventions(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot negative intervention results.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plot_curves(
        plot_results,
        plot_key,
        groupby=groupby,
        title=f"Negative Interventions: {format_plot_title(plot_key)} {name}",
        x_label="Fraction of Concepts Intervened",
        y_label="Classification Error",
        get_x=lambda results: np.linspace(
            0, 1, len(results[0].metrics["neg_intervention_accs"]["y"])
        ),
        get_y=lambda result: 1 - result.metrics["neg_intervention_accs"]["y"],
        eval_mode="neg_intervention",
        save_dir=save_dir,
        save_name="neg_intervention",
        prefix=name,
        show=show,
    )


def plot_positive_interventions(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot positive intervention results.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plot_curves(
        plot_results,
        plot_key,
        groupby=groupby,
        title=f"Positive Interventions: {format_plot_title(plot_key)} {name}",
        x_label="Fraction of Concepts Intervened",
        y_label="Classification Accuracy",
        get_x=lambda results: np.linspace(
            0, 1, len(results[0].metrics["pos_intervention_accs"]["y"])
        ),
        get_y=lambda result: result.metrics["pos_intervention_accs"]["y"],
        eval_mode="pos_intervention",
        save_dir=save_dir,
        save_name="pos_intervention",
        prefix=name,
        show=show,
    )


def plot_threshold_fitting(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot positive intervention results.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plot_curves(
        plot_results,
        plot_key,
        groupby=groupby,
        title=f"Setting Thresshold: {format_plot_title(plot_key)} {name}",
        x_label="Thresshold",
        y_label="Classification Accuracy",
        get_x=lambda results: np.linspace(
            0.4, 0.6, len(results[0].metrics["threshold_fitting"]["y"])
        ),
        get_y=lambda result: result.metrics["threshold_fitting"]["y"],
        eval_mode="threshold_fitting",
        save_dir=save_dir,
        save_name="threshold_fitting",
        prefix=name,
        show=show,
    )


def plot_random_concepts_residual(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot results with randomized concepts and residuals.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plt.clf()
    save_path = get_save_path(plot_key, prefix=name, suffix="random", save_dir=save_dir)

    baseline_accs, baseline_stds = [], []
    random_concept_accs, random_concept_stds = [], []
    random_residual_accs, random_residual_stds = [], []

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    info = (
        (baseline_accs, baseline_stds, "accuracy", "test_acc"),
        (
            random_concept_accs,
            random_concept_stds,
            "random_concepts",
            "random_concept_acc",
        ),
        (
            random_residual_accs,
            random_residual_stds,
            "random_residual",
            "random_residual_acc",
        ),
    )
    for key in plot_results.keys():
        results = group_results(plot_results[key], groupby="eval_mode")
        for values, stds, eval_mode, metric in info:
            values.append(
                np.mean([result.metrics[metric] for result in results[eval_mode]])
            )
            stds.append(
                np.std([result.metrics[metric] for result in results[eval_mode]])
            )

    # Create CSV file
    data = np.stack(
        [
            baseline_accs,
            baseline_stds,
            random_concept_accs,
            random_concept_stds,
            random_residual_accs,
            random_residual_stds,
        ],
        axis=1,
    )
    columns = [
        "Baseline Acc.",
        "Baseline Std.",
        "Random Concepts Acc.",
        "Random Concepts Std.",
        "Random Residual Acc.",
        "Random Residual Std.",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Create figure
    x = np.arange(len(plot_results.keys()))
    plt.bar(x - 0.25, baseline_accs, label="Baseline", width=0.25)
    plt.bar(x, random_concept_accs, label="Random Concepts", width=0.25)
    plt.bar(x + 0.25, random_residual_accs, label="Random Residual", width=0.25)
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Classification Accuracy")
    plt.title(f"Random Concepts & Residual: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_disentanglement(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot disentanglement metrics.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    x_metric, y_metric = "mean_abs_cross_correlation", "mutual_info"
    plot_scatter(
        plot_results,
        plot_key,
        groupby=groupby,
        title=f"Disentanglement Metrics: {format_plot_title(plot_key)} {name}",
        x_label="Mean Absolute Cross Correlation",
        y_label="Mutual Information",
        x_eval_mode="correlation",
        y_eval_mode="mutual_info",
        get_x=lambda results: np.mean([result.metrics[x_metric] for result in results]),
        get_y=lambda results: np.mean([result.metrics[y_metric] for result in results]),
        save_dir=save_dir,
        save_name="disentanglement",
        prefix=name,
        show_regression_line=False,
        show=show,
    )


def plot_mutual_info(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["decorrelation_type"],
    residual_dim_key: str = "residual_dim",
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot mutual information scores with residual dimension on x-axis and bars for each method type.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by (default is decorrelation_type)
    residual_dim_key : str
        The key for residual dimension in the results
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Additional name identifier for the plot
    """
    plt.clf()
    save_path = get_save_path(plot_key, prefix=name, suffix="mi", save_dir=save_dir)

    # Extract groupby key (usually decorrelation_type or method)
    groupby_key = (
        groupby[0] if isinstance(groupby, list) and len(groupby) > 0 else groupby
    )

    # Get all unique residual dimensions
    residual_dims = sorted(
        set(result.config["residual_dim"] for result in plot_results)
    )

    # Get all unique method types (e.g., decorrelation types)
    method_types = sorted(set(result.config[groupby_key] for result in plot_results))

    # Create data structure to hold results by dimension and method
    # Structure: {residual_dim: {method_type: [mi_scores]}}
    results_data = {
        dim: {method: [] for method in method_types} for dim in residual_dims
    }

    # Group and aggregate results
    for result in plot_results:
        res_dim = result.config[residual_dim_key]
        method = result.config[groupby_key]

        # Add the MI score if it exists
        if "mutual_info" in result.metrics:
            results_data[res_dim][method].append(result.metrics["mutual_info"])

    # Calculate means and standard deviations
    means_data = {
        dim: {
            method: np.mean(scores) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in results_data.items()
    }

    stds_data = {
        dim: {
            method: np.std(scores) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in results_data.items()
    }

    # Prepare data for CSV
    csv_data = []
    for dim in residual_dims:
        for method in method_types:
            csv_data.append(
                [dim, method, means_data[dim][method], stds_data[dim][method]]
            )

    # Create the figure
    fig, ax = plt.subplots(figsize=(max(8, len(residual_dims) * 2), 6))

    # Set bar width based on number of methods
    bar_width = 0.8 / len(method_types)

    import matplotlib.cm as cm

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    method_colors = {
        method: colors[i % len(colors)] for i, method in enumerate(method_types)
    }

    # Plot bars for each residual dimension and method
    for i, dim in enumerate(residual_dims):
        for j, method in enumerate(method_types):
            # Calculate x position
            x_pos = i + (j - len(method_types) / 2 + 0.5) * bar_width

            # Get data
            mean_val = means_data[dim][method]
            std_val = stds_data[dim][method]

            # Plot bar (only add label for first residual dimension)
            ax.bar(
                x_pos,
                mean_val,
                width=bar_width,
                yerr=std_val,
                color=method_colors[method],
                label=str(method) if i == 0 else None,
            )

    # Set x-axis
    ax.set_xticks(range(len(residual_dims)))
    ax.set_xticklabels([str(dim) for dim in residual_dims])
    ax.set_xlabel("Residual Dimension")
    ax.set_ylabel("Mutual Information Score")
    ax.set_title(
        f"Mutual Information by Dimension: {format_plot_title(plot_key)} {name}"
    )

    # Add legend
    ax.legend(loc="best")

    # Create and save CSV
    columns = [
        "Residual Dimension",
        f"{groupby_key.replace('_', ' ').title()}",
        "MI Score Mean",
        "MI Score Std",
    ]
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Save and show
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_intervention_vs_disentanglement(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot intervention accuracy metrics vs disentanglement metrics.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    x_info = [
        (
            "Mean Absolute Cross Correlation",
            "correlation",
            "mean_abs_cross_correlation",
        ),
        ("Mutual Information", "mutual_info", "mutual_info"),
    ]
    y_info = [
        ("Positive Intervention Accuracy", "pos_intervention", "pos_intervention_accs"),
    ]
    for x_label, x_eval_mode, x_metric in x_info:
        for y_label, y_eval_mode, y_metric in y_info:
            if y_metric == "pos_intervention_accs":
                get_y = lambda results: np.stack(
                    [result.metrics[y_metric]["y"][-1] for result in results]
                )
            else:
                get_y = lambda results: np.stack(
                    [result.metrics[y_metric] for result in results]
                )
            plot_scatter(
                plot_results,
                plot_key,
                groupby=groupby,
                title=f"{y_label} vs. {x_label}\n{format_plot_title(plot_key)} {name}",
                x_label=x_label,
                y_label=y_label,
                x_eval_mode=x_eval_mode,
                y_eval_mode=y_eval_mode,
                get_x=lambda results: np.stack(
                    [result.metrics[x_metric] for result in results]
                ),
                get_y=get_y,
                save_dir=save_dir,
                save_name=f"{y_metric}_vs_{x_metric}",
                prefix=name,
                show_regression_line=True,
                show=show,
            )


def plot_concept_predictions(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_hidden_concepts: bool = False,
):
    """
    Plot results for concept disentanglement.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plt.clf()
    save_path_disentanglement = get_save_path(
        plot_key, prefix=name, suffix="disentanglement", save_dir=save_dir
    )
    save_path_change = get_save_path(
        plot_key, prefix=name, suffix="change", save_dir=save_dir
    )

    supervised_accs, hidden_accs = [], []
    supervised_change, hidden_change = [], []
    supervised_accs_std, hidden_accs_std = [], []
    supervised_change_std, hidden_change_std = [], []

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    info = (
        (supervised_accs, supervised_accs_std, "concept_pred", 0),
        (supervised_change, supervised_change_std, "concept_pred", 2),
    )
    if plot_hidden_concepts:
        info += (
            (hidden_accs, hidden_accs_std, "concept_pred", 1),
            (hidden_change, hidden_change_std, "concept_pred", 3),
        )
    for key in plot_results.keys():
        results = group_results(plot_results[key], groupby="eval_mode")
        for values, stds, eval_mode, metric_idx in info:
            values.append(
                np.mean(
                    [
                        result.metrics[eval_mode][metric_idx]
                        for result in results[eval_mode]
                    ]
                )
            )
            stds.append(
                np.std(
                    [
                        result.metrics[eval_mode][metric_idx]
                        for result in results[eval_mode]
                    ]
                )
            )

    # Create CSV file
    if plot_hidden_concepts:
        data = np.stack(
            [
                supervised_accs,
                hidden_accs,
                supervised_change,
                hidden_change,
            ],
            axis=1,
        )
        columns = [
            "Supervised Concepts Acc.",
            "Hidden Concepts Acc.",
            "Supervised Concepts Change",
            "Hidden Concepts Change",
        ]
    else:
        data = np.stack(
            [
                supervised_accs,
                supervised_change,
            ],
            axis=1,
        )
        columns = [
            "Supervised Concepts Acc.",
            "Supervised Concepts Change",
        ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path_disentanglement.with_suffix(".csv"), index=False)

    # Create figure for concept accuracy
    x = np.arange(len(plot_results.keys()))
    plt.figure()
    plt.bar(
        x - 0.2,
        supervised_accs,
        yerr=supervised_accs_std,
        label="Supervised Concepts Acc.",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x,
            hidden_accs,
            yerr=hidden_accs_std,
            label="Hidden Concepts Acc.",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Accuracy")
    plt.title(f"Concept Accuracy: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_disentanglement.with_suffix(".pdf"))
    if show:
        plt.show()

    # Create figure for concept prediction change
    plt.figure()
    plt.bar(
        x - 0.2,
        supervised_change,
        yerr=supervised_change_std,
        label="Supervised Concepts Change",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x,
            hidden_change,
            yerr=hidden_change_std,
            label="Hidden Concepts Change",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Change")
    plt.title(f"Concept Prediction Change: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_change.with_suffix(".pdf"))
    if show:
        plt.show()
    print("here")


def plot_all_concept_predictions(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["decorrelation_type"],
    residual_dim_key: str = "residual_dim",
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot mutual information scores with residual dimension on x-axis and bars for each method type.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by (default is decorrelation_type)
    residual_dim_key : str
        The key for residual dimension in the results
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Additional name identifier for the plot
    """
    plt.clf()
    save_path = get_save_path(
        plot_key, prefix=name, suffix="concept_pred_acc", save_dir=save_dir
    )

    # Extract groupby key (usually decorrelation_type or method)
    groupby_key = (
        groupby[0] if isinstance(groupby, list) and len(groupby) > 0 else groupby
    )

    # Get all unique residual dimensions
    residual_dims = sorted(
        set(result.config["residual_dim"] for result in plot_results)
    )

    # Get all unique method types (e.g., decorrelation types)
    method_types = sorted(set(result.config[groupby_key] for result in plot_results))

    # Create data structure to hold results by dimension and method
    # Structure: {residual_dim: {method_type: [mi_scores]}}
    results_data = {
        dim: {method: [] for method in method_types} for dim in residual_dims
    }

    individual_f1_data = {
        dim: {method: [] for method in method_types} for dim in residual_dims
    }
    # Group and aggregate results
    for result in plot_results:
        res_dim = result.config[residual_dim_key]
        method = result.config[groupby_key]

        # Add the MI score if it exists
        if "concept_pred" in result.metrics:
            results_data[res_dim][method].append(result.metrics["concept_pred"][0])
            individual_f1_data[res_dim][method].append(
                result.metrics["concept_pred"][1]
            )

    individual_f1_data_means = {
        dim: {
            method: np.mean(np.stack(scores), axis=0) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in individual_f1_data.items()
    }
    individual_f1_data_stds = {
        dim: {
            method: np.std(np.stack(scores), axis=0) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in individual_f1_data.items()
    }
    individual_means_csv_path = get_save_path(
        plot_key, prefix=name, suffix="concept_pred_acc_means", save_dir=save_dir
    ).with_suffix(".csv")
    individual_stds_csv_path = get_save_path(
        plot_key, prefix=name, suffix="concept_pred_acc_means_stds", save_dir=save_dir
    ).with_suffix(".csv")

    # Prepare means data for CSV
    means_rows = []
    stds_rows = []
    for dim in residual_dims:
        for method in method_types:
            # Create row for means
            means_row = {
                "Residual Dimension": dim,
                f"{groupby_key.replace('_', ' ').title()}": method,
            }

            # Create row for stds
            stds_row = {
                "Residual Dimension": dim,
                f"{groupby_key.replace('_', ' ').title()}": method,
            }

            # Add each concept's mean and std
            if isinstance(individual_f1_data_means[dim][method], np.ndarray):
                for i, score in enumerate(individual_f1_data_means[dim][method]):
                    means_row[f"concept_{i}"] = score

            if isinstance(individual_f1_data_stds[dim][method], np.ndarray):
                for i, score in enumerate(individual_f1_data_stds[dim][method]):
                    stds_row[f"concept_{i}"] = score

            means_rows.append(means_row)
            stds_rows.append(stds_row)

    # Save to CSVs
    pd.DataFrame(means_rows).to_csv(individual_means_csv_path, index=False)
    pd.DataFrame(stds_rows).to_csv(individual_stds_csv_path, index=False)

    # Calculate means and standard deviations
    means_data = {
        dim: {
            method: np.mean(scores) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in results_data.items()
    }

    stds_data = {
        dim: {
            method: np.std(scores) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in results_data.items()
    }

    # Prepare data for CSV
    csv_data = []
    for dim in residual_dims:
        for method in method_types:
            csv_data.append(
                [dim, method, means_data[dim][method], stds_data[dim][method]]
            )

    # Create the figure
    fig, ax = plt.subplots(figsize=(max(8, len(residual_dims) * 2), 6))

    # Set bar width based on number of methods
    bar_width = 0.8 / len(method_types)

    import matplotlib.cm as cm

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    method_colors = {
        method: colors[i % len(colors)] for i, method in enumerate(method_types)
    }

    # Plot bars for each residual dimension and method
    for i, dim in enumerate(residual_dims):
        for j, method in enumerate(method_types):
            # Calculate x position
            x_pos = i + (j - len(method_types) / 2 + 0.5) * bar_width

            # Get data
            mean_val = means_data[dim][method]
            std_val = stds_data[dim][method]

            # Plot bar (only add label for first residual dimension)
            ax.bar(
                x_pos,
                mean_val,
                width=bar_width,
                yerr=std_val,
                color=method_colors[method],
                label=str(method) if i == 0 else None,
            )

    # Set x-axis
    ax.set_xticks(range(len(residual_dims)))
    ax.set_xticklabels([str(dim) for dim in residual_dims])
    ax.set_xlabel("Residual Dimension")
    ax.set_ylabel("Concept Prediction F1 Score")
    ax.set_title(
        f"Concept Prediction F1 Score from the Residual Layer: {format_plot_title(plot_key)} {name}"
    )

    # Add legend
    ax.legend(loc="best")

    # Create and save CSV
    columns = [
        "Residual Dimension",
        f"{groupby_key.replace('_', ' ').title()}",
        "Concept Acc Mean",
        "Concept Acc Std",
    ]
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Save and show
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_all_counterfactual_intervention_2(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["decorrelation_type"],
    residual_dim_key: str = "residual_dim",
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot mutual information scores with residual dimension on x-axis and bars for each method type.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by (default is decorrelation_type)
    residual_dim_key : str
        The key for residual dimension in the results
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Additional name identifier for the plot
    """
    plt.clf()
    save_path = get_save_path(
        plot_key,
        prefix=name,
        suffix="counterfactual_intervention_single",
        save_dir=save_dir,
    )

    # Extract groupby key (usually decorrelation_type or method)
    groupby_key = (
        groupby[0] if isinstance(groupby, list) and len(groupby) > 0 else groupby
    )

    # Get all unique residual dimensions
    residual_dims = sorted(
        set(
            result.config["residual_dim"]
            for result in plot_results
            if (result is not None) and (result.config is not None)
        )
    )

    # Get all unique method types (e.g., decorrelation types)
    method_types = sorted(
        set(
            result.config[groupby_key]
            for result in plot_results
            if (result is not None) and (result.config is not None)
        )
    )

    # Create data structure to hold results by dimension and method
    # Structure: {residual_dim: {method_type: [mi_scores]}}
    results_data = {
        dim: {method: [] for method in method_types} for dim in residual_dims
    }

    # Group and aggregate results
    for result in plot_results:
        if result is None or result.config is None:
            continue
        res_dim = result.config[residual_dim_key]
        method = result.config[groupby_key]

        # Add the MI score if it exists
        if "test_counterfactual_2" in result.metrics:
            results_data[res_dim][method].append(
                result.metrics["test_counterfactual_2"][0]
            )

    # Calculate means and standard deviations
    means_data = {
        dim: {
            method: np.mean(scores) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in results_data.items()
    }

    stds_data = {
        dim: {
            method: np.std(scores) if scores else np.nan
            for method, scores in methods.items()
        }
        for dim, methods in results_data.items()
    }

    # Prepare data for CSV
    csv_data = []
    for dim in residual_dims:
        for method in method_types:
            csv_data.append(
                [dim, method, means_data[dim][method], stds_data[dim][method]]
            )

    # Create the figure
    fig, ax = plt.subplots(figsize=(max(8, len(residual_dims) * 2), 6))

    # Set bar width based on number of methods
    bar_width = 0.8 / len(method_types)

    import matplotlib.cm as cm

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    method_colors = {
        method: colors[i % len(colors)] for i, method in enumerate(method_types)
    }

    # Plot bars for each residual dimension and method
    for i, dim in enumerate(residual_dims):
        for j, method in enumerate(method_types):
            # Calculate x position
            x_pos = i + (j - len(method_types) / 2 + 0.5) * bar_width

            # Get data
            mean_val = means_data[dim][method]
            std_val = stds_data[dim][method]

            # Plot bar (only add label for first residual dimension)
            ax.bar(
                x_pos,
                mean_val,
                width=bar_width,
                yerr=std_val,
                color=method_colors[method],
                label=str(method) if i == 0 else None,
            )

    # Set x-axis
    ax.set_xticks(range(len(residual_dims)))
    ax.set_xticklabels([str(dim) for dim in residual_dims])
    ax.set_xlabel("Residual Dimension")
    ax.set_ylabel("Counterfactual intervention Accuracy")
    ax.set_title(
        f"Counterfactual: Intervene on white concept should change brown bear prediction to Polar: {format_plot_title(plot_key)} {name}"
    )

    # Add legend
    ax.legend(loc="best")

    # Create and save CSV
    columns = [
        "Residual Dimension",
        f"{groupby_key.replace('_', ' ').title()}",
        "Counterfactual Acc Mean",
        "Counterfactual Acc Std",
    ]
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Save and show
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_all_counterfactual_intervention(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["decorrelation_type"],
    residual_dim_key: str = "residual_dim",
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot histograms showing class distributions before and after intervention.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by (default is decorrelation_type)
    residual_dim_key : str
        The key for residual dimension in the results
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Additional name identifier for the plot
    """
    save_path = get_save_path(
        plot_key, prefix=name, suffix="counterfactual_intervention", save_dir=save_dir
    )

    # Extract groupby key (usually decorrelation_type or method)
    groupby_key = (
        groupby[0] if isinstance(groupby, list) and len(groupby) > 0 else groupby
    )

    # Get all unique residual dimensions
    residual_dims = sorted(
        set(
            result.config["residual_dim"]
            for result in plot_results
            if (result is not None) and (result.config is not None)
        )
    )

    # Get all unique method types (e.g., decorrelation types)
    method_types = sorted(
        set(
            result.config[groupby_key]
            for result in plot_results
            if (result is not None) and (result.config is not None)
        )
    )

    # Create data structure to hold class distributions by dimension and method
    # Structure: {residual_dim: {method_type: {original_classes: [], predicted_classes: []}}}
    results_data = {
        dim: {
            method: {"original_classes": [], "predicted_classes": []}
            for method in method_types
        }
        for dim in residual_dims
    }

    # Group and aggregate results
    for result in plot_results:
        if result is None or result.config is None:
            continue
        res_dim = result.config[residual_dim_key]
        method = result.config[groupby_key]

        # Add the class distributions if they exist
        if "test_counterfactual" in result.metrics:
            # Assuming test_counterfactual now returns (original_classes, predicted_classes)
            original_classes, predicted_classes = result.metrics["test_counterfactual"]
            results_data[res_dim][method]["original_classes"].extend(original_classes)
            results_data[res_dim][method]["predicted_classes"].extend(predicted_classes)

    # Prepare color palette
    import matplotlib.cm as cm

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    method_colors = {
        method: colors[i % len(colors)] for i, method in enumerate(method_types)
    }

    # Create a figure for each residual dimension
    for dim in residual_dims:
        # Create the figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Class Distribution Before and After White Concept Intervention\nResidual Dimension: {dim} - {format_plot_title(plot_key)} {name}",
            fontsize=16,
        )

        # Find all unique class indices across all methods for this dimension
        all_original_classes = []
        all_predicted_classes = []
        for method in method_types:
            all_original_classes.extend(results_data[dim][method]["original_classes"])
            all_predicted_classes.extend(results_data[dim][method]["predicted_classes"])

        unique_classes = sorted(set(all_original_classes + all_predicted_classes))

        # Create histograms for each method
        bar_width = 0.8 / len(method_types)

        # Plot original classes (before intervention)
        ax1.set_title("Before Intervention (Original Classes)")
        for j, method in enumerate(method_types):
            original_classes = results_data[dim][method]["original_classes"]
            if original_classes:
                # Count occurrences of each class
                class_counts = {}
                for cls in unique_classes:
                    class_counts[cls] = original_classes.count(cls)

                # Plot bars for each class
                x_positions = (
                    np.arange(len(unique_classes))
                    + (j - len(method_types) / 2 + 0.5) * bar_width
                )
                heights = [class_counts.get(cls, 0) for cls in unique_classes]

                ax1.bar(
                    x_positions,
                    heights,
                    width=bar_width,
                    color=method_colors[method],
                    label=str(method),
                )

        # Plot predicted classes (after intervention)
        ax2.set_title("After Intervention (Predicted Classes)")
        for j, method in enumerate(method_types):
            predicted_classes = results_data[dim][method]["predicted_classes"]
            if predicted_classes:
                # Count occurrences of each class
                class_counts = {}
                for cls in unique_classes:
                    class_counts[cls] = predicted_classes.count(cls)

                # Plot bars for each class
                x_positions = (
                    np.arange(len(unique_classes))
                    + (j - len(method_types) / 2 + 0.5) * bar_width
                )
                heights = [class_counts.get(cls, 0) for cls in unique_classes]

                ax2.bar(
                    x_positions,
                    heights,
                    width=bar_width,
                    color=method_colors[method],
                    label=method,
                )

        # Set x-axis labels and ticks
        for ax in [ax1, ax2]:
            ax.set_xticks(np.arange(len(unique_classes)))

            # Try to map class indices to class names if possible
            # This assumes there's a dataset in the first result with class_to_idx mapping
            if len(plot_results) > 0:
                first_result = next(iter(plot_results))
                if hasattr(first_result, "dataset") and hasattr(
                    first_result.dataset, "animals_class_to_idx"
                ):
                    idx_to_class = {
                        v: k
                        for k, v in first_result.dataset.animals_class_to_idx.items()
                    }
                    class_labels = [
                        idx_to_class.get(cls, str(cls)) for cls in unique_classes
                    ]
                else:
                    class_labels = [str(cls) for cls in unique_classes]
            else:
                class_labels = [str(cls) for cls in unique_classes]

            ax.set_xticklabels(class_labels, rotation=45, ha="right")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")

        # Add legend to the figure
        ax1.legend(loc="upper right")

        # Adjust layout and save
        plt.tight_layout()
        residual_dim_subfolder = save_path.parent / f"residual_dim_{dim}"
        residual_dim_subfolder.mkdir(parents=True, exist_ok=True)
        plt.savefig(residual_dim_subfolder / f"{save_path.stem}_dim_{dim}.pdf")

        if show:
            plt.show()

        plt.close(fig)

    # Create a CSV with summary data
    csv_data = []
    for dim in residual_dims:
        for method in method_types:
            original_classes = results_data[dim][method]["original_classes"]
            predicted_classes = results_data[dim][method]["predicted_classes"]

            # Skip if no data
            if not original_classes or not predicted_classes:
                continue

            # Calculate shift statistics
            total_samples = len(original_classes)
            unchanged = sum(
                1 for o, p in zip(original_classes, predicted_classes) if o == p
            )
            changed = total_samples - unchanged

            # Add to CSV data
            csv_data.append(
                [
                    dim,
                    method,
                    total_samples,
                    unchanged,
                    changed,
                    changed / total_samples if total_samples > 0 else 0,
                ]
            )

    # Create DataFrame and save CSV
    columns = [
        "Residual Dimension",
        f"{groupby_key.replace('_', ' ').title()}",
        "Total Samples",
        "Unchanged Classes",
        "Changed Classes",
        "Proportion Changed",
    ]
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)


def plot_confusion_matrix(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["decorrelation_type"],
    residual_dim_key: str = "residual_dim",
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot individual confusion matrices for each residual dimension and model type.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by (default is decorrelation_type)
    residual_dim_key : str
        The key for residual dimension in the results
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Additional name identifier for the plot
    """
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Create the base save path using the same function as the original code
    base_save_path = get_save_path(
        plot_key, prefix=name, suffix="confusion_matrices", save_dir=save_dir
    )

    # Extract groupby key (usually decorrelation_type or method)
    groupby_key = (
        groupby[0] if isinstance(groupby, list) and len(groupby) > 0 else groupby
    )

    # Get all unique residual dimensions
    residual_dims = sorted(
        set(result.config["residual_dim"] for result in plot_results)
    )

    # Get all unique method types (e.g., decorrelation types)
    method_types = sorted(set(result.config[groupby_key] for result in plot_results))

    # Create data structure to hold confusion matrices by dimension and method
    confusion_matrices = {
        dim: {method: [] for method in method_types} for dim in residual_dims
    }

    # Group and collect confusion matrices
    for result in plot_results:
        res_dim = result.config[residual_dim_key]
        method = result.config[groupby_key]

        # Add the confusion matrix if it exists
        if "test_confusion_matrix" in result.metrics:
            confusion_matrices[res_dim][method].append(
                result.metrics["test_confusion_matrix"]
            )

    # Create CSV data to collect results
    csv_data = []

    # Plot each confusion matrix individually
    for dim in residual_dims:
        for method in method_types:
            matrices = confusion_matrices[dim][method]

            if not matrices:  # Skip if no data
                continue

            # Average the matrices if there are multiple runs
            if len(matrices) > 1:
                avg_matrix = np.mean(matrices, axis=0)
                std_matrix = np.std(matrices, axis=0)
            else:
                avg_matrix = matrices[0]
                std_matrix = np.zeros_like(avg_matrix)

            # Create filename for this specific combination using the base save path
            save_path = base_save_path.with_name(
                f"{base_save_path.stem}_dim{dim}_{method}"
            )

            # Determine number of classes in the matrix
            num_classes = avg_matrix.shape[0]

            # Plot the confusion matrix
            plt.figure(figsize=(12, 10) if num_classes > 10 else (8, 6))

            # For large matrices (like 50x50), don't show individual annotations
            if num_classes <= 10:
                class_names = [f"Class {i}" for i in range(num_classes)]
                sns.heatmap(
                    avg_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                )
            else:
                # For large matrices, skip annotations and use fewer ticks
                sns.heatmap(avg_matrix, annot=False, cmap="Blues")
                # Add ticks at regular intervals
                tick_interval = max(1, num_classes // 10)  # Show at most 10 ticks
                plt.xticks(
                    np.arange(0, num_classes, tick_interval) + 0.5,
                    [f"{i}" for i in range(0, num_classes, tick_interval)],
                )
                plt.yticks(
                    np.arange(0, num_classes, tick_interval) + 0.5,
                    [f"{i}" for i in range(0, num_classes, tick_interval)],
                )

            plt.title(f"Confusion Matrix\nDim: {dim}, Method: {method}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()

            # Save the plot
            plt.savefig(save_path.with_suffix(".pdf"))
            plt.savefig(save_path.with_suffix(".png"))

            if show:
                plt.show()
            plt.close()

            # Calculate metrics from confusion matrix
            if num_classes == 2:  # Binary classification
                tn, fp, fn, tp = avg_matrix.ravel()
                accuracy = (
                    (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                )
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                # Add metrics to CSV data
                csv_data.append(
                    [dim, method, accuracy, precision, recall, f1, tp, fp, fn, tn]
                )
            else:  # Multiclass
                # Calculate accuracy
                accuracy = (
                    np.sum(np.diag(avg_matrix)) / np.sum(avg_matrix)
                    if np.sum(avg_matrix) > 0
                    else 0
                )

                # Calculate class-wise metrics
                precision_list = []
                recall_list = []
                f1_list = []

                for i in range(num_classes):
                    tp = avg_matrix[i, i]
                    fp = np.sum(avg_matrix[:, i]) - tp
                    fn = np.sum(avg_matrix[i, :]) - tp

                    # Calculate metrics with zero handling
                    prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec_i = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_i = (
                        2 * prec_i * rec_i / (prec_i + rec_i)
                        if (prec_i + rec_i) > 0
                        else 0
                    )

                    precision_list.append(prec_i)
                    recall_list.append(rec_i)
                    f1_list.append(f1_i)

                # Calculate macro averages
                macro_precision = np.mean(precision_list)
                macro_recall = np.mean(recall_list)
                macro_f1 = np.mean(f1_list)

                # Add metrics to CSV data
                csv_data.append(
                    [
                        dim,
                        method,
                        accuracy,
                        macro_precision,
                        macro_recall,
                        macro_f1,
                        "N/A",  # Not applicable for multiclass
                        "N/A",
                        "N/A",
                        "N/A",
                    ]
                )

                # For large matrices, add a separate visualization for top classes
                if num_classes > 10:
                    plt.figure(figsize=(12, 6))

                    # Get diagonal elements (correct predictions)
                    diag = np.diag(avg_matrix)

                    # Get class frequencies
                    class_freq = np.sum(avg_matrix, axis=1)

                    # Get indices of top classes by frequency
                    top_n = min(20, num_classes)
                    top_indices = np.argsort(class_freq)[-top_n:][
                        ::-1
                    ]  # Top n, descending

                    # Plot top classes
                    plt.bar(range(top_n), diag[top_indices])
                    plt.title(
                        f"Accuracy per class (top {top_n} classes)\nDim: {dim}, Method: {method}"
                    )
                    plt.xlabel("Class Index")
                    plt.ylabel("Correctly Classified Samples")
                    plt.xticks(range(top_n), [str(i) for i in top_indices])

                    # Add class accuracy percentages
                    for i, idx in enumerate(top_indices):
                        acc = diag[idx] / class_freq[idx] if class_freq[idx] > 0 else 0
                        plt.text(i, diag[idx] + 0.5, f"{acc:.2f}", ha="center")

                    plt.tight_layout()
                    accuracy_path = save_path.with_name(
                        f"{save_path.stem}_class_accuracy"
                    )
                    plt.savefig(accuracy_path.with_suffix(".pdf"))
                    plt.savefig(accuracy_path.with_suffix(".png"))
                    plt.close()

    # Create and save CSV with metrics
    columns = [
        "Residual Dimension",
        f"{groupby_key.replace('_', ' ').title()}",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "True Positives",
        "False Positives",
        "False Negatives",
        "True Negatives",
    ]

    metrics_path = base_save_path.with_name(
        f"{base_save_path.stem}_metrics"
    ).with_suffix(".csv")
    pd.DataFrame(csv_data, columns=columns).to_csv(metrics_path, index=False)

    print(f"Saved confusion matrices and metrics to {save_dir}")


def plot_residual_pca_scatter(
    plot_results: ResultGrid,
    plot_key: str | tuple[str],
    groupby: list[str] = ["decorrelation_type"],
    residual_dim_key: str = "residual_dim",
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
):
    """
    Plot PCA scatter plots showing residual distributions for grizzly bears and polar bears.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : str
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by (default is decorrelation_type)
    residual_dim_key : str
        The key for residual dimension in the results
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Additional name identifier for the plot
    """
    save_path = get_save_path(
        plot_key, prefix=name, suffix="residual_pca_analysis", save_dir=save_dir
    )

    # Extract groupby key (usually decorrelation_type or method)
    groupby_key = (
        groupby[0] if isinstance(groupby, list) and len(groupby) > 0 else groupby
    )

    # Get all unique residual dimensions
    residual_dims = sorted(
        set(result.config["residual_dim"] for result in plot_results)
    )

    # Get all unique method types (e.g., decorrelation types)
    method_types = sorted(set(result.config[groupby_key] for result in plot_results))

    # Create data structure to hold PCA points by dimension and method
    # Structure: {residual_dim: {method_type: {grizzly_points: [], polar_points: []}}}
    results_data = {
        dim: {
            method: {"grizzly_points": [], "polar_points": []}
            for method in method_types
        }
        for dim in residual_dims
    }

    # Group and aggregate results
    for result in plot_results:
        res_dim = result.config[residual_dim_key]
        method = result.config[groupby_key]

        # Add the PCA points if they exist
        if "pca" in result.metrics:
            # Assuming analyze_residuals_with_pca returns (grizzly_points, polar_points)
            grizzly_points, polar_points = result.metrics["pca"]
            # breakpoint()
            results_data[res_dim][method]["grizzly_points"] = grizzly_points
            results_data[res_dim][method]["polar_points"] = polar_points

    # Create a figure for each method type and residual dimension combination
    for dim in residual_dims:
        num_methods = len(method_types)
        fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 6))

        # Handle case with only one method
        if num_methods == 1:
            axes = [axes]

        fig.suptitle(
            f"PCA of Residuals: Grizzly Bear vs Polar Bear\nResidual Dimension: {dim} - {format_plot_title(plot_key)} {name}",
            fontsize=16,
        )

        for i, (method, ax) in enumerate(zip(method_types, axes)):
            grizzly_points = results_data[dim][method]["grizzly_points"]
            polar_points = results_data[dim][method]["polar_points"]

            if not grizzly_points or not polar_points:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{method}")
                continue

            # Extract x and y coordinates
            grizzly_x = [p[0] for p in grizzly_points]
            grizzly_y = [p[1] for p in grizzly_points]

            polar_x = [p[0] for p in polar_points]
            polar_y = [p[1] for p in polar_points]

            # Plot the scatter points
            ax.scatter(
                grizzly_x, grizzly_y, color="brown", alpha=0.7, label="Grizzly Bear"
            )
            ax.scatter(polar_x, polar_y, color="skyblue", alpha=0.7, label="Polar Bear")

            # Add labels and legend
            ax.set_title(f"{method}")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend()
            ax.grid(alpha=0.3)

            # Optional: Calculate and display overlap statistics
            if grizzly_points and polar_points:
                # Calculate simple overlap metric (using convex hulls or distance metrics)
                # This is a placeholder for more sophisticated metrics
                from scipy.spatial import ConvexHull
                import numpy as np

                try:
                    if len(grizzly_points) >= 3 and len(polar_points) >= 3:
                        grizzly_array = np.array(grizzly_points)
                        polar_array = np.array(polar_points)

                        grizzly_hull = ConvexHull(grizzly_array)
                        polar_hull = ConvexHull(polar_array)

                        # Calculate centroids
                        grizzly_centroid = np.mean(grizzly_array, axis=0)
                        polar_centroid = np.mean(polar_array, axis=0)

                        # Calculate distance between centroids
                        centroid_distance = np.linalg.norm(
                            grizzly_centroid - polar_centroid
                        )

                        # Add annotation with distance
                        ax.text(
                            0.05,
                            0.95,
                            f"Centroid distance: {centroid_distance:.2f}",
                            transform=ax.transAxes,
                            fontsize=9,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", alpha=0.1),
                        )
                except:
                    # Skip if there's an error with convex hull calculation
                    pass

        # Adjust layout and save
        plt.tight_layout()
        residual_dim_subfolder = save_path.parent / f"residual_dim_{dim}"
        residual_dim_subfolder.mkdir(parents=True, exist_ok=True)
        plt.savefig(residual_dim_subfolder / f"{save_path.stem}_dim_{dim}.pdf")

        if show:
            plt.show()

        plt.close(fig)

    # Create a CSV with summary data
    csv_data = []
    for dim in residual_dims:
        for method in method_types:
            grizzly_points = results_data[dim][method]["grizzly_points"]
            polar_points = results_data[dim][method]["polar_points"]

            # Skip if no data
            if not grizzly_points or not polar_points:
                continue

            # Calculate statistics
            num_grizzly = len(grizzly_points)
            num_polar = len(polar_points)

            # Calculate centroid distance if possible
            if num_grizzly > 0 and num_polar > 0:
                import numpy as np

                grizzly_array = np.array(grizzly_points)
                polar_array = np.array(polar_points)

                grizzly_centroid = np.mean(grizzly_array, axis=0)
                polar_centroid = np.mean(polar_array, axis=0)

                centroid_distance = np.linalg.norm(grizzly_centroid - polar_centroid)
            else:
                centroid_distance = float("nan")

            # Add to CSV data
            csv_data.append([dim, method, num_grizzly, num_polar, centroid_distance])

    # Create DataFrame and save CSV
    columns = [
        "Residual Dimension",
        f"{groupby_key.replace('_', ' ').title()}",
        "Grizzly Bear Samples",
        "Polar Bear Samples",
        "Centroid Distance",
    ]

    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(save_path.with_suffix(".csv"), index=False)


def plot_concept_change(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_hidden_concepts: bool = False,
):
    """
    Plot results for concept change.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    plot_hidden_concepts : bool
        Whether to plot hidden concepts
    """
    plt.clf()
    save_path_change = get_save_path(
        plot_key, prefix=name, suffix="concept_change", save_dir=save_dir
    )

    num_changed_concepts, concept_updated_when_wrong, hidden_concepts_updated = (
        [],
        [],
        [],
    )
    (
        num_changed_concepts_std,
        concept_updated_when_wrong_std,
        hidden_concepts_updated_std,
    ) = (
        [],
        [],
        [],
    )

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    for key in plot_results.keys():
        results = plot_results[key]
        num_changed_concepts.append(
            np.mean([result.metrics["concept_change"][0] for result in results])
        )
        concept_updated_when_wrong.append(
            np.mean([result.metrics["concept_change"][1] for result in results])
        )
        hidden_concepts_updated.append(
            np.mean([result.metrics["concept_change"][2] for result in results])
        )
        num_changed_concepts_std.append(
            np.std([result.metrics["concept_change"][0] for result in results])
        )
        concept_updated_when_wrong_std.append(
            np.std([result.metrics["concept_change"][1] for result in results])
        )
        hidden_concepts_updated_std.append(
            np.std([result.metrics["concept_change"][2] for result in results])
        )

    # Create CSV file
    data = np.stack(
        [
            num_changed_concepts,
            concept_updated_when_wrong,
            hidden_concepts_updated,
            num_changed_concepts_std,
            concept_updated_when_wrong_std,
            hidden_concepts_updated_std,
        ],
        axis=1,
    )
    columns = [
        "Num Changed Concepts",
        "Concept Updated When Wrong",
        "Hidden Concepts Updated",
        "Num Changed Concepts Std.",
        "Concept Updated When Wrong Std.",
        "Hidden Concepts Updated Std.",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path_change.with_suffix(".csv"), index=False)

    # Create figure for concept change
    x = np.arange(len(plot_results.keys()))
    plt.figure()
    plt.bar(
        x - 0.2,
        num_changed_concepts,
        yerr=num_changed_concepts_std,
        label="Num Changed Concepts",
        width=0.2,
        capsize=5,
    )
    plt.bar(
        x,
        concept_updated_when_wrong,
        yerr=concept_updated_when_wrong_std,
        label="Concept Updated",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x + 0.2,
            hidden_concepts_updated,
            yerr=hidden_concepts_updated_std,
            label="Hidden Concepts Updated",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Metrics")
    plt.title(f"Concept Change: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_change.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_concept_change_probe(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_hidden_concepts: bool = False,
):
    """
    Plot results for concept change probe.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    plot_hidden_concepts : bool
        Whether to plot hidden concepts
    """
    plt.clf()
    save_path_change_probe = get_save_path(
        plot_key, prefix=name, suffix="concept_change_probe", save_dir=save_dir
    )

    accuracy, num_changed_concepts, concept_updated, hidden_concepts_updated = (
        [],
        [],
        [],
        [],
    )
    (
        accuracy_std,
        num_changed_concepts_std,
        concept_updated_std,
        hidden_concepts_updated_std,
    ) = ([], [], [], [])

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    for key in plot_results.keys():
        results = plot_results[key]
        accuracy.append(
            np.mean([result.metrics["concept_change_probe"][0] for result in results])
        )
        num_changed_concepts.append(
            np.mean([result.metrics["concept_change_probe"][1] for result in results])
        )
        concept_updated.append(
            np.mean([result.metrics["concept_change_probe"][2] for result in results])
        )
        hidden_concepts_updated.append(
            np.mean([result.metrics["concept_change_probe"][3] for result in results])
        )
        accuracy_std.append(
            np.std([result.metrics["concept_change_probe"][0] for result in results])
        )
        num_changed_concepts_std.append(
            np.std([result.metrics["concept_change_probe"][1] for result in results])
        )
        concept_updated_std.append(
            np.std([result.metrics["concept_change_probe"][2] for result in results])
        )
        hidden_concepts_updated_std.append(
            np.std([result.metrics["concept_change_probe"][3] for result in results])
        )

    # Create CSV file
    data = np.stack(
        [
            num_changed_concepts,
            concept_updated,
            hidden_concepts_updated,
            num_changed_concepts_std,
            concept_updated_std,
            hidden_concepts_updated_std,
        ],
        axis=1,
    )
    columns = [
        "Accuracy",
        "Num Changed Concepts",
        "Concept Updated",
        "Hidden Concepts Updated",
        "Num Changed Concepts Std.",
        "Concept Updated Std.",
        "Hidden Concepts Updated Std.",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path_change_probe.with_suffix(".csv"), index=False)

    # Create figure for concept change probe
    x = np.arange(len(plot_results.keys()))
    plt.figure()
    plt.bar(
        x - 0.3,
        accuracy,
        yerr=accuracy_std,
        label="Accuracy",
        width=0.2,
        capsize=5,
    )
    plt.bar(
        x - 0.1,
        num_changed_concepts,
        yerr=num_changed_concepts_std,
        label="Num Changed Concepts",
        width=0.2,
        capsize=5,
    )
    plt.bar(
        x + 0.1,
        concept_updated,
        yerr=concept_updated_std,
        label="Concept Updated",
        width=0.2,
        capsize=5,
    )
    if plot_hidden_concepts:
        plt.bar(
            x + 0.3,
            hidden_concepts_updated,
            yerr=hidden_concepts_updated_std,
            label="Hidden Concepts Updated",
            width=0.2,
            capsize=5,
        )
    plt.xticks(np.arange(len(plot_results.keys())), plot_results.keys())
    plt.ylabel("Metrics")
    plt.title(f"Concept Change Probe: {format_plot_title(plot_key)} {name}")
    plt.legend()
    plt.savefig(save_path_change_probe.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_tcav(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_magnitude: bool = False,
):
    """
    Plot results for concept change.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    plot_hidden_concepts : bool
        Whether to plot hidden concepts
    """
    plt.clf()
    pm = "_magnitude" if plot_magnitude else ""
    save_path = get_save_path(
        plot_key, prefix=name, suffix=f"tcav{pm}", save_dir=save_dir
    )

    def tcav_process(result, subkey):
        """
        Extracts the TCAV scores from the result.
        """
        if plot_magnitude:
            # Use magnitude instead of sign_count for the plot
            metric_to_use = "magnitude"
        else:
            metric_to_use = "sign_count"
        # metric_to_use = "magnitude"
        return {
            key: (
                np.mean(value[metric_to_use][subkey]),
                np.std(value[metric_to_use][subkey])
                / np.sqrt(len(value[metric_to_use][subkey])),
            )
            for key, value in result.metrics["tcav_scores"].items()
        }

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    all_scores = {}
    for key in plot_results.keys():
        results = plot_results[key]
        tcav_scores = [
            tcav_process(result, "P1")
            for result in results
            if "tcav_scores" in result.metrics
        ]
        all_scores[key] = tcav_scores

    all_concepts = set()
    for method, scores_list in all_scores.items():
        for scores in scores_list:
            all_concepts.update(scores.keys())

    all_concepts = sorted(list(all_concepts))
    methods = sorted(list(all_scores.keys()))

    # Create figure with one subplot per concept
    # n_concepts = len(all_concepts)
    # fig, axes = plt.subplots(1, n_concepts, figsize=(5 * n_concepts, 8), sharey=True)

    # # If there's only one concept, axes won't be an array
    # if n_concepts == 1:
    #     axes = [axes]

    # # Plot each concept
    # for i, concept in enumerate(all_concepts):
    #     ax = axes[i]

    #     # Set up bar positions with spacing for each method
    #     bar_width = 0.8 / len(methods)
    #     method_positions = np.arange(len(methods))

    #     # Plot each method
    #     for j, method in enumerate(methods):
    #         means = []
    #         stds = []
    #         for score_dict in all_scores[method]:
    #             if concept in score_dict:
    #                 mean, std = score_dict[concept]
    #                 print(concept, score_dict[concept])
    #                 means.append(mean)
    #                 stds.append(std)

    #         if means:
    #             # Calculate the position for each bar with appropriate spacing
    #             x_positions = method_positions[j]

    #             # Plot mean as a ba
    #             print(stds)
    #             ax.bar(
    #                 x_positions,
    #                 np.mean(means),  # np.abs(0.5 - np.mean(means)),
    #                 width=bar_width,
    #                 yerr=np.mean(stds),  # Standard error
    #                 capsize=5,
    #                 color=plt.cm.tab10.colors[j],
    #                 edgecolor="black",
    #                 alpha=0.7,
    #                 label=str(method),
    #             )

    #             # Add individual data points if there are multiple results
    #             if len(means) > 1:
    #                 scatter_positions = np.random.normal(
    #                     x_positions, bar_width / 8, len(means)
    #                 )
    #                 ax.scatter(scatter_positions, means, color="black", s=20, zorder=3)

    #     # Set title and labels
    #     ax.set_title(f"{concept}", fontsize=14)
    #     if i == 0:
    #         ax.set_ylabel("TCAV Score")
    #     ax.grid(axis="y", linestyle="--", alpha=0.3)

    #     # Set x-axis labels
    #     ax.set_xticks(method_positions)
    #     ax.set_xticklabels(methods, rotation=45, ha="right")

    # # Add legend in the last subplot only if there are multiple methods
    # if len(methods) > 1:
    #     handles, labels = axes[-1].get_legend_handles_labels()
    #     if handles:
    #         fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    # plt.tight_layout()
    # plt.suptitle("TCAV Scores Across Methods", fontsize=16, y=1.02)
    # plt.savefig(save_path.with_suffix(".pdf"))

    # if show:
    #     plt.show()
    n_concepts = len(all_concepts)

    # Calculate grid dimensions - max 4 concepts per row
    max_cols = 4
    n_cols = min(n_concepts, max_cols)
    n_rows = (n_concepts + max_cols - 1) // max_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False
    )

    # If there's only one row, axes might not be a 2D array
    if n_rows == 1:
        if n_cols == 1:
            axes = np.array([[axes]])
        else:
            axes = np.array([axes])

    # Plot each concept
    for i, concept in enumerate(all_concepts):
        # Calculate row and column for this concept
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Set up bar positions with spacing for each method
        bar_width = 0.8 / len(methods)
        method_positions = np.arange(len(methods))

        # Plot each method
        for j, method in enumerate(methods):
            means = []
            stds = []
            for score_dict in all_scores[method]:
                if concept in score_dict:
                    mean, std = score_dict[concept]
                    print(concept, score_dict[concept])
                    means.append(mean)
                    stds.append(std)

            if means:
                # Calculate the position for each bar with appropriate spacing
                x_positions = method_positions[j]

                # Plot mean as a bar
                print(stds)
                ax.bar(
                    x_positions,
                    np.mean(means),  # np.abs(0.5 - np.mean(means)),
                    width=bar_width,
                    yerr=np.mean(stds),
                    capsize=5,
                    color=plt.cm.tab10.colors[
                        j % 10
                    ],  # Cycle through colors if more than 10 methods
                    edgecolor="black",
                    alpha=0.7,
                    label=str(method),
                )

                # Add individual data points if there are multiple results
                if len(means) > 1:
                    scatter_positions = np.random.normal(
                        x_positions, bar_width / 8, len(means)
                    )
                    ax.scatter(scatter_positions, means, color="black", s=20, zorder=3)
        all_values = []
        for m_idx, m_name in enumerate(methods):
            for score_dict in all_scores[m_name]:
                if concept in score_dict:
                    mean, std = score_dict[concept]
                    all_values.append(mean)
                    # Include error bar values
                    all_values.append(mean + std)
                    all_values.append(
                        max(0, mean - std)
                    )  # Prevent negative values if not meaningful

        if all_values:
            # Set y-axis limits with 10% padding
            min_val = min(all_values)
            max_val = max(all_values)
            y_range = max_val - min_val

            # If range is very small, use a minimum range to avoid tiny plots
            if y_range < 0.05:
                padding = 0.05
            else:
                padding = y_range * 0.15  # 15% padding

            # Set limits, ensuring we don't go below 0 if values are all positive
            y_min = max(0, min_val - padding) if min_val >= 0 else min_val - padding
            y_max = max_val + padding

            ax.set_ylim(y_min, y_max)
        # Set title and labels
        ax.set_title(f"{concept}", fontsize=14)
        if col == 0:  # Only set y-label for leftmost plots
            ax.set_ylabel("TCAV Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Set x-axis labels
        ax.set_xticks(method_positions)
        ax.set_xticklabels(methods, rotation=45, ha="right")

    # Hide unused subplots
    for i in range(n_concepts, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    # Add legend only once for the entire figure
    if len(methods) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.suptitle("TCAV Scores Across Methods", fontsize=16, y=1.02)
    plt.savefig(save_path.with_suffix(".pdf"))

    if show:
        plt.show()


def plot_attribution(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_concepts: bool = False,
):
    """
    Plot results for concept change.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    plot_hidden_concepts : bool
        Whether to plot hidden concepts
    """
    plt.clf()
    which_attribution = "concept" if plot_concepts else "residual"
    make_attr_folder = save_dir / "attribution"
    make_attr_folder.mkdir(parents=True, exist_ok=True)
    save_path = get_save_path(
        plot_key,
        prefix=name,
        suffix=f"deeplift_shapley_{which_attribution}",
        save_dir=make_attr_folder,
    )
    csv_save_path = save_path.with_suffix(".csv")

    raw_save_path = get_save_path(
        plot_key,
        prefix=name,
        suffix=f"deeplift_shapley_{which_attribution}_raw",
        save_dir=make_attr_folder,
    )
    raw_save_path = get_save_path(
        plot_key,
        prefix=name,
        suffix=f"deeplift_shapley_{which_attribution}_raw",
        save_dir=make_attr_folder,
    )
    raw_save_path = save_path.with_suffix(".pkl")

    # if True or not os.path.exists(raw_save_path):
    #     just_shap = [
    #         result.metrics
    #         for result in plot_results
    #         if "deeplift_shapley" in result.metrics
    #     ]
    #     import pickle

    #     with open(raw_save_path, "wb") as f:
    #         pickle.dump(just_shap, f)

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    all_scores = {}
    all_scores_2 = {}
    for key in plot_results.keys():
        results = plot_results[key]
        attribution_scores = [
            result.metrics["deeplift_shapley"][f"{which_attribution}_attributions"]
            for result in results
            if "deeplift_shapley" in result.metrics
        ]
        try:
            attribution_scores_ = np.stack(attribution_scores, axis=0)
            attribution_scores_2 = np.concatenate(attribution_scores, axis=0)
        except:
            breakpoint()
        all_scores[key] = attribution_scores_2
        all_scores_2[key] = attribution_scores_

    # Get dimensions from first entry
    first_array = next(iter(all_scores.values()))
    n_concepts = first_array.shape[1]
    methods = list(all_scores.keys())

    # Aggregate results
    #groupby = groupby[0] if len(groupby) == 1 else groupby
    #plot_results = group_results(plot_results, groupby=groupby)
    # all_scores = {}
    # all_scores_2 = {}
    # for key in plot_results.keys():
    #     results = plot_results[key]
    #     attribution_scores = [
    #         result.metrics["deeplift_shapley"][f"{which_attribution}_attributions"]
    #         for result in results
    #         if "deeplift_shapley" in result.metrics
    #     ]
    #     try:
    #         attribution_scores_ = np.stack(attribution_scores, axis=0)
    #         attribution_scores_2 = np.concatenate(attribution_scores, axis=0)
    #     except:
    #         breakpoint()
    #     all_scores[key] = attribution_scores_2
    #     all_scores_2[key] = attribution_scores_

    # # Get dimensions from first entry
    # first_array = next(iter(all_scores.values()))
    # n_concepts = first_array.shape[1]
    # methods = list(all_scores.keys())

    # Calculate and save average attributions to CSV
    import pandas as pd

    # # Create a DataFrame to store average attributions
    # avg_attributions = {}
    # for method, scores in all_scores.items():
    #     # Calculate mean across all samples for each concept
    #     avg_attributions[method] = np.mean(scores, axis=0)

    # # Convert to DataFrame
    # avg_df = pd.DataFrame(
    #     avg_attributions, index=[f"concept_{i}" for i in range(n_concepts)]
    # )
    avg_attributions = {}
    for method, scores in all_scores_2.items():
        # Calculate mean across all samples for each concept
        t = np.mean(scores, axis=1)
        t = np.mean(t, axis=1)

        avg_attributions[method] = (np.mean(t), np.std(t))

    # # Convert to DataFrame
    avg_df = pd.DataFrame(avg_attributions, index=["mean", "std"])

    # Save to CSV
    avg_df.to_csv(csv_save_path)
    print(f"Average attributions saved to {csv_save_path}")

    # # Calculate grid dimensions

    # max_cols = 4
    # n_cols = min(n_concepts, max_cols)
    # n_rows = (n_concepts + max_cols - 1) // max_cols
    # if n_rows == 0 or n_cols == 0:
    #     return

    # # Create figure
    # fig, axes = plt.subplots(
    #     n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False
    # )

    # # Handle axes array formatting
    # if n_rows == 1 and n_cols == 1:
    #     axes = np.array([[axes]])
    # elif n_rows == 1:
    #     axes = np.array([axes])
    # elif n_cols == 1:
    #     axes = axes.reshape(-1, 1)

    # # Plot each concept dimension
    # for concept_idx in range(n_concepts):
    #     row = concept_idx // n_cols
    #     col = concept_idx % n_cols
    #     ax = axes[row, col]

    #     # Collect all values for bin range calculation
    #     all_values = np.concatenate(
    #         [scores[:, concept_idx] for scores in all_scores.values()]
    #     )

    #     # Calculate bins (auto or custom logic)
    #     bins = 20  # Could use np.histogram_bin_edges(all_values, bins='auto')

    #     # Plot histograms for each method
    #     for method_idx, (method, scores) in enumerate(all_scores.items()):
    #         ax.hist(
    #             scores[:, concept_idx],
    #             bins=bins,
    #             alpha=0.7,
    #             density=True,
    #             label=method,
    #             color=plt.cm.tab10(method_idx % 10),
    #             edgecolor="black",
    #             linewidth=0.5,
    #         )

    #     # Format subplot
    #     ax.set_title(f"Concept {concept_idx}", fontsize=12)
    #     ax.set_xlabel("Attribution Score")
    #     if col == 0:
    #         ax.set_ylabel("Density")
    #     ax.grid(alpha=0.3)
    #     ax.set_axisbelow(True)

    # # Clean up empty subplots
    # for i in range(n_concepts, n_rows * n_cols):
    #     row = i // n_cols
    #     col = i % n_cols
    #     fig.delaxes(axes[row, col])

    # # Add legend
    # if len(methods) > 1:
    #     handles, labels = axes[0, 0].get_legend_handles_labels()
    #     fig.legend(
    #         handles,
    #         labels,
    #         loc="upper center",
    #         ncol=min(4, len(methods)),
    #         bbox_to_anchor=(0.5, 1.02),
    #     )

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.92)
    # plt.suptitle("Concept Attribution Distributions", fontsize=16)
    # plt.savefig(save_path, bbox_inches="tight")

    # if show:
    #     plt.show()


def plot_posthoc_cbm_performance(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    crm: bool = True,
    no_cbm: bool = False,
):
    """
    Plot results with number of concepts, baseline and intervention accuracy.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    """
    plt.clf()
    crm = "_cr" if crm else ""
    if no_cbm:
        no_cbm_name = "_no_cbm"
        key_name = "posthoc_res"
    else:
        no_cbm_name = ""
        key_name = f"posthoc_fitter{crm}"
    save_path = get_save_path(
        plot_key,
        prefix=name,
        suffix=f"intervention{crm}{no_cbm_name}",
        save_dir=save_dir,
    )

    # Extract groupby key (usually decorrelation_type or method)
    groupby_key = (
        groupby[0] if isinstance(groupby, list) and len(groupby) > 0 else groupby
    )

    # Get all unique residual dimensions
    residual_dims = sorted(
        set(
            result.config["residual_dim"]
            for result in plot_results
            if result is not None
            and result.config is not None
            and key_name in result.metrics
        )
    )

    # Get all unique method types (e.g., decorrelation types)
    method_types = sorted(
        set(
            result.config[groupby_key]
            for result in plot_results
            if result is not None
            and result.config is not None
            and key_name in result.metrics
        )
    )

    # Aggregate results
    grouped_results = group_results(plot_results, groupby=groupby_key)

    # For each group (x-axis category)
    keys = list(grouped_results.keys())
    num_concepts_list, baseline_accs, intervention_accs = [], [], []

    for key in keys:
        # Extract values from posthoc_fitter at positions 0, 1, and 2
        results = grouped_results[key]
        avg_num_concepts = np.mean(
            [
                result.metrics[key_name][0]
                for result in results
                if result is not None
                and result.config is not None
                and f"posthoc_fitter{crm}" in result.metrics
            ]
        )
        avg_baseline = np.mean(
            [
                result.metrics[key_name][1]
                for result in results
                if result is not None
                and result.config is not None
                and key_name in result.metrics
            ]
        )
        avg_intervention = np.mean(
            [
                result.metrics[key_name][2]
                for result in results
                if result is not None
                and result.config is not None
                and key_name in result.metrics
            ]
        )

        num_concepts_list.append(avg_num_concepts)
        baseline_accs.append(avg_baseline)
        intervention_accs.append(avg_intervention)

    # Create CSV file with detailed information
    # First collect data by residual dimension and method type
    data_by_dim_method = {
        dim: {
            method: {"num_concepts": [], "baseline": [], "intervention": []}
            for method in method_types
        }
        for dim in residual_dims
    }

    for result in plot_results:
        if (
            result is not None
            or result.config is not None
            or key_name in result.metrics
        ):
            continue
        dim = result.config["residual_dim"]
        method = result.config[groupby_key]

        if "posthoc_fitter" in result.metrics:
            data_by_dim_method[dim][method]["num_concepts"].append(
                result.metrics[key_name][0]
            )
            data_by_dim_method[dim][method]["baseline"].append(
                result.metrics[key_name][1]
            )
            data_by_dim_method[dim][method]["intervention"].append(
                result.metrics[key_name][2]
            )

    # Create DataFrame for detailed CSV
    rows = []
    for dim in residual_dims:
        for method in method_types:
            data = data_by_dim_method[dim][method]
            if data["num_concepts"]:  # Only add if there's data
                rows.append(
                    {
                        "residual_dim": dim,
                        groupby_key: method,
                        "num_concepts": np.mean(data["num_concepts"]),
                        "baseline_accuracy": np.mean(data["baseline"]),
                        "intervention_accuracy": np.mean(data["intervention"]),
                    }
                )

    detailed_df = pd.DataFrame(rows)
    detailed_df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Create summary DataFrame for the plot
    summary_df = pd.DataFrame(
        {
            groupby_key: keys,
            "num_concepts": num_concepts_list,
            "baseline_accuracy": baseline_accs,
            "intervention_accuracy": intervention_accs,
        }
    )
    summary_df.to_csv(
        save_path.with_name(f"{save_path.stem}_summary").with_suffix(".csv"),
        index=False,
    )

    # Create figure
    x = np.arange(len(keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    # Scale num_concepts to be visible on same plot as accuracies
    max_concepts = max(num_concepts_list) if num_concepts_list else 1
    scaled_concepts = [n / max_concepts for n in num_concepts_list]

    # Create bars
    bar1 = ax.bar(
        x - width,
        scaled_concepts,
        width,
        label=f"Num Concepts (max={max_concepts:.1f})",
    )
    bar2 = ax.bar(x, baseline_accs, width, label="Baseline Accuracy")
    bar3 = ax.bar(x + width, intervention_accs, width, label="Intervention Accuracy")

    # Add second y-axis for actual concept numbers
    ax2 = ax.twinx()
    ax2.set_ylabel("Number of Concepts")
    # ax2.set_ylim(0, max_concepts * 1.1)  # 10% margin

    # Add labels and styling
    ax.set_xlabel(f"Grouped by {groupby_key}")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Performance Comparison: {format_plot_title(plot_key)} {name}")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".pdf"))
    if show:
        plt.show()


def plot_mean_attribution_old(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_concepts: bool = False,
    top_n: int = 20,  # Number of top concepts to show when there are many
):
    """
    Plot mean attribution scores for each concept as lines.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Name prefix for saved file
    plot_concepts : bool
        Whether to plot concept attributions (True) or residual attributions (False)
    top_n : int
        Maximum number of concepts to show individually when there are many
    """
    plt.clf()
    which_attribution = "concept" if plot_concepts else "residual"
    make_attr_folder = save_dir / "attribution"
    make_attr_folder.mkdir(parents=True, exist_ok=True)
    save_path = get_save_path(
        plot_key,
        prefix=name,
        suffix=f"deeplift_shapley_{which_attribution}_mean",
        save_dir=make_attr_folder,
    )

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    all_mean_scores = {}

    for key in plot_results.keys():
        results = plot_results[key]
        attribution_scores = [
            result.metrics["deeplift_shapley"][f"{which_attribution}_attributions"]
            for result in results
            if "deeplift_shapley" in result.metrics
        ]

        if attribution_scores:
            # Concatenate all attribution scores
            all_attributions = np.concatenate(attribution_scores, axis=0)
            # Calculate mean across all samples for each concept
            mean_scores = np.mean(all_attributions, axis=0)
            all_mean_scores[key] = mean_scores

    if not all_mean_scores:
        print("No attribution data found")
        return

    # Get dimensions
    first_array = next(iter(all_mean_scores.values()))
    n_concepts = first_array.shape[0]
    methods = list(all_mean_scores.keys())

    # Create figure with appropriate size
    plt.figure(figsize=(12, 6))

    # # Get global sorting order by averaging absolute mean scores across all methods
    # global_mean = np.zeros(n_concepts)
    # for mean_scores in all_mean_scores.values():
    #     global_mean += np.abs(mean_scores)
    # global_mean /= len(all_mean_scores)
    # sorted_indices = np.argsort(-global_mean)

    # # Keep only top_n concepts if there are many
    # if n_concepts > top_n:
    #     sorted_indices = sorted_indices[:top_n]

    # Plot each method as a line on the same plot
    for method_idx, (method, mean_scores) in enumerate(all_mean_scores.items()):
        # Sort according to the global sorting
        sorted_indices = np.argsort(-np.abs(mean_scores))
        sorted_indices = sorted_indices[:top_n]
        sorted_scores = mean_scores[sorted_indices]

        # Plot as line with markers
        plt.plot(
            range(len(sorted_scores)),
            sorted_scores,
            marker="o",
            linewidth=2,
            markersize=6,
            label=method,
            color=plt.cm.tab10(method_idx % 10),
            alpha=0.8,
        )

    # Set plot parameters
    plt.ylabel("Mean Attribution Score")
    plt.title(
        f"Mean {which_attribution.capitalize()} Attribution Scores\n(Sorted by Average Absolute Magnitude)"
    )
    plt.grid(alpha=0.3)

    # Hide x-tick labels but keep the ticks
    plt.xticks(range(len(sorted_indices)))

    # Add minimal tick lines to indicate concept positions
    plt.tick_params(axis="x", which="both", length=4, width=1, direction="out", pad=8)

    # Add note about truncation if needed
    if n_concepts > top_n:
        plt.figtext(
            0.5,
            0.01,
            f"Note: Only showing top {top_n} concepts out of {n_concepts} total (sorted by absolute value)",
            ha="center",
            fontsize=10,
            style="italic",
        )

    # Add legend
    plt.legend(loc="best")

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()


def plot_mean_attribution(
    plot_results: ResultGrid,
    plot_key: tuple[str, ...],
    groupby: list[str] = ["model_type"],
    save_dir: Path | str = "./plots",
    show: bool = True,
    name: str = "",
    plot_concepts: bool = False,
    top_n: int = 20,  # Number of top concepts to show when there are many
    save_stats_csv: bool = True,  # Whether to save stats to CSV
):
    """
    Plot mean attribution scores for each concept as lines and save statistics to CSV.

    Parameters
    ----------
    plot_results : ResultGrid
        Results for the given plot
    plot_key : tuple[str]
        Identifier for this plot
    groupby : list[str]
        List of train config keys to group by
    save_dir : Path or str
        Directory to save plots to
    show : bool
        Whether to show the plot
    name : str
        Name prefix for saved file
    plot_concepts : bool
        Whether to plot concept attributions (True) or residual attributions (False)
    top_n : int
        Maximum number of concepts to show individually when there are many
    save_stats_csv : bool
        Whether to save summary statistics to CSV
    """
    plt.clf()
    which_attribution = "concept" if plot_concepts else "residual"
    make_attr_folder = save_dir / "attribution"
    make_attr_folder.mkdir(parents=True, exist_ok=True)
    save_path = get_save_path(
        plot_key,
        prefix=name,
        suffix=f"deeplift_shapley_{which_attribution}_mean",
        save_dir=make_attr_folder,
    )

    # Aggregate results
    groupby = groupby[0] if len(groupby) == 1 else groupby
    plot_results = group_results(plot_results, groupby=groupby)
    all_mean_scores = {}

    # For CSV export
    csv_data = []

    for key in plot_results.keys():
        results = plot_results[key]
        attribution_scores = [
            result.metrics["deeplift_shapley"][f"{which_attribution}_attributions"]
            for result in results
            if "deeplift_shapley" in result.metrics
        ]

        if attribution_scores:
            # Concatenate all attribution scores
            all_attributions = np.concatenate(attribution_scores, axis=0)
            batch_attributions = np.stack(attribution_scores, axis=0)

            # Calculate mean across all concepts for each sample (axis=1)
            batch_means = np.mean(batch_attributions, axis=1)
            batch_means = np.mean(batch_means, axis=1)

            # Calculate mean and std of the sample means
            batch_mean = np.mean(batch_means)
            batch_std = np.std(batch_means)

            # Store for CSV export
            model_type = plot_key[0]
            csv_data.append(
                {
                    "Residual Dimension": key,
                    "Model Type": model_type,
                    "Shap Mean": batch_mean,
                    "Shap Std": batch_std,
                }
            )

            # Calculate mean across all samples for each concept (for plotting)
            mean_scores = np.mean(all_attributions, axis=0)
            all_mean_scores[key] = mean_scores

    if not all_mean_scores:
        print("No attribution data found")
        return

    # Save statistics to CSV
    if save_stats_csv and csv_data:
        import pandas as pd

        stats_df = pd.DataFrame(csv_data)
        stats_df.to_csv(save_path.with_suffix(".csv"), index=False)

    # Get dimensions
    first_array = next(iter(all_mean_scores.values()))
    n_concepts = first_array.shape[0]
    methods = list(all_mean_scores.keys())

    # Create figure with appropriate size
    plt.figure(figsize=(12, 6))

    # Plot each method as a line on the same plot
    for method_idx, (method, mean_scores) in enumerate(all_mean_scores.items()):
        # Sort according to the global sorting
        sorted_indices = np.argsort(-np.abs(mean_scores))
        sorted_indices = sorted_indices[:top_n]
        sorted_scores = mean_scores[sorted_indices]

        # Plot as line with markers
        plt.plot(
            range(len(sorted_scores)),
            sorted_scores,
            marker="o",
            linewidth=2,
            markersize=6,
            label=method,
            color=plt.cm.tab10(method_idx % 10),
            alpha=0.8,
        )

    # Set plot parameters
    plt.ylabel("Mean Attribution Score")
    plt.title(
        f"Mean {which_attribution.capitalize()} Attribution Scores\n(Sorted by Average Absolute Magnitude)"
    )
    plt.grid(alpha=0.3)

    # Hide x-tick labels but keep the ticks
    plt.xticks(range(len(sorted_indices)))

    # Add minimal tick lines to indicate concept positions
    plt.tick_params(axis="x", which="both", length=4, width=1, direction="out", pad=8)

    # Add note about truncation if needed
    if n_concepts > top_n:
        plt.figtext(
            0.5,
            0.01,
            f"Note: Only showing top {top_n} concepts out of {n_concepts} total (sorted by absolute value)",
            ha="center",
            fontsize=10,
            style="italic",
        )

    # Add legend
    plt.legend(loc="best")

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()


if __name__ == "__main__":
    PLOT_FUNCTIONS = {
        # "neg_intervention": plot_negative_interventions,
        # "pos_intervention": plot_positive_interventions,
        # "random": plot_random_concepts_residual,
        # "concept_pred": plot_concept_predictions,
        #"all_concept_pred": plot_all_concept_predictions,
        # "counterfactual_intervention": plot_all_counterfactual_intervention,
        "counterfactual_intervention_single": plot_all_counterfactual_intervention_2,
        # "plot_residual_pca_scatter": plot_residual_pca_scatter,
        # "plot_confusion_matrix": plot_confusion_matrix,
        # "concept_change": plot_concept_change,
        # "concept_change_probe": plot_concept_change_probe,
        # "concept_change": plot_concept_changes,
        # "disentanglement": plot_disentanglement,
        # "intervention_vs_disentanglement": plot_intervention_vs_disentanglement,
        # "tcav_sign_count": partial(plot_tcav, plot_magnitude=False),
        # "tcav_magnitude": partial(plot_tcav, plot_magnitude=True),
        #"attribution": plot_attribution,
        # "attribution_concepts": partial(plot_attribution, plot_concepts=True),
        # "attribution_residuals": partial(plot_attribution, plot_concepts=False),
        # "mean_attribution_concepts": partial(plot_mean_attribution, plot_concepts=True),
        # "mean_attribution_residuals": partial(
        #     plot_mean_attribution, plot_concepts=False
        # ),
        # "plot_threshold_fitting": plot_threshold_fitting,
        # "mutual_info": plot_mutual_info,
        # "posthoc_cbm": plot_posthoc_cbm_performance,
        # "posthoc_crm": partial(plot_posthoc_cbm_performance, no_cbm=True),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        help="Experiment directory",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=PLOT_FUNCTIONS.keys(),
        help="Plot modes",
    )
    parser.add_argument(
        "--plotby",
        nargs="+",
        default=["dataset"],
        help=(
            "Config keys to group plots by "
            "(e.g. `--plotby dataset model_type` creates separate plots "
            "for each (dataset, model_type) combination)"
        ),
    )
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["model_type"],
        help="Config keys to group results on each plot by",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to show the plot(s)",
    )

    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    folder = "eval"
    experiment_paths = Path(args.exp_dir).resolve().glob(f"**/{folder}/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load evaluation results
    print("Loading evaluation results from", experiment_path / folder)
    tuner = tune.Tuner.restore(str(experiment_path / folder), trainable=evaluate)
    results = group_results(tuner.get_results(), groupby=args.plotby)

    # Plot results
    plot_folder = "plots"
    for plot_key, plot_results in tqdm(results.items()):
        for mode in tqdm(args.mode):
            PLOT_FUNCTIONS[mode](
                plot_results,
                plot_key,
                groupby=args.groupby,
                save_dir=experiment_path / plot_folder,
                show=args.show,
            )
