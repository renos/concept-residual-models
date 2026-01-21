"""
Plot concept prediction F1 vs SHAP importance scatter plot (Figure 3 from paper).

This script visualizes the relationship between how well concepts can be predicted
from the residual (F1 score) and how important those concepts are for the task
(SHAP value).

Usage:
    python scripts/plot_concept_vs_shap.py --exp-dir <EXPERIMENT_DIR>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import linregress


METHOD_NAMES = {
    "latent_residual": "Latent Residual",
    "decorrelated_residual": "Decorrelated Residual",
    "iter_norm": "Iterative Norm",
    "eye": "EYE",
    "adversarial_decorrelation": "Adversarial Decorrelation",
    "mi_residual": "MI Residual",
}


def load_concept_shap_results(exp_dir: Path):
    """Load concept prediction and SHAP results from experiment directory."""
    results = {}
    plots_dir = exp_dir / "plots"

    if not plots_dir.exists():
        raise FileNotFoundError(f"Plots directory not found: {plots_dir}")

    # Look for per-concept CSV files
    pattern = "*per_concept*.csv"
    files = list(plots_dir.glob(pattern))

    for f in files:
        df = pd.read_csv(f)
        # Try to infer method from filename
        method = None
        for m in METHOD_NAMES.keys():
            if m in f.name.lower():
                method = m
                break
        if method:
            results[method] = df

    return results


def plot_concept_vs_shap(
    results: dict,
    output_path: Path,
    dataset_name: str = None,
):
    """Plot F1 score vs SHAP value for each concept."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))

    if len(results) == 1:
        axes = [axes]

    for ax, (method, df) in zip(axes, results.items()):
        # Look for F1 and SHAP columns
        f1_col = None
        shap_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "f1" in col_lower:
                f1_col = col
            elif "shap" in col_lower or "importance" in col_lower:
                shap_col = col

        if f1_col is None or shap_col is None:
            continue

        x = df[shap_col].values
        y = df[f1_col].values

        # Plot scatter
        ax.scatter(x, y, alpha=0.7, s=50)

        # Add regression line
        if len(x) > 2:
            slope, intercept, r_value, _, _ = linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', alpha=0.7)
            ax.text(0.05, 0.95, f'$R^2$ = {r_value**2:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_xlabel("SHAP Importance", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(METHOD_NAMES.get(method, method), fontsize=12)

    title = "Concept Prediction F1 vs SHAP Importance"
    if dataset_name:
        title = f"{dataset_name}: {title}"
    fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot concept prediction F1 vs SHAP importance")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for figure")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for title")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    results = load_concept_shap_results(exp_dir)

    if not results:
        print(f"No concept vs SHAP results found in {exp_dir}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = exp_dir / "concept_prediction_vs_shap.pdf"

    plot_concept_vs_shap(results, output_path, args.dataset)


if __name__ == "__main__":
    main()
