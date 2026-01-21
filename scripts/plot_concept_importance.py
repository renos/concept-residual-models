"""
Plot concept importance (SHAP) figures (Figure 6 from paper).

Usage:
    python scripts/plot_concept_importance.py --exp-dir <EXPERIMENT_DIR>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter


METHOD_NAMES = {
    "latent_residual": "Latent Residual",
    "decorrelated_residual": "Decorrelated Residual",
    "iter_norm": "Iterative Norm",
    "eye": "EYE",
    "adversarial_decorrelation": "Adversarial Decorrelation",
    "mi_residual": "MI Residual",
}

METHOD_ORDER = [
    "latent_residual",
    "decorrelated_residual",
    "iter_norm",
    "eye",
    "adversarial_decorrelation",
    "mi_residual",
]


def load_shap_results(exp_dir: Path):
    """Load SHAP importance results from experiment directory."""
    results = {}
    plots_dir = exp_dir / "plots"

    if not plots_dir.exists():
        raise FileNotFoundError(f"Plots directory not found: {plots_dir}")

    # Look for CSV files with SHAP results
    for method in METHOD_ORDER:
        pattern = f"*{method}*shap*.csv"
        files = list(plots_dir.glob(pattern))
        if not files:
            pattern = f"*{method}*importance*.csv"
            files = list(plots_dir.glob(pattern))
        if files:
            results[method] = pd.read_csv(files[0])

    return results


def plot_concept_importance(
    results: dict,
    output_path: Path,
    dataset_name: str = None,
):
    """Plot DeepLIFT SHAP concept importance."""
    plt.figure(figsize=(8, 5))

    for method, df in results.items():
        if method not in METHOD_NAMES:
            continue

        # Look for SHAP/importance columns
        importance_col = None
        std_col = None
        for col in df.columns:
            if "shap" in col.lower() or "importance" in col.lower():
                if "std" in col.lower():
                    std_col = col
                else:
                    importance_col = col

        if importance_col is None:
            continue

        x = np.arange(len(df))
        if std_col:
            plt.errorbar(x, df[importance_col], yerr=df[std_col],
                        fmt='-o', label=METHOD_NAMES[method])
        else:
            plt.plot(x, df[importance_col], '-o', label=METHOD_NAMES[method])

    plt.xlabel("Residual Dimension", fontsize=12)
    plt.ylabel("Concept Importance (SHAP)", fontsize=12)

    title = "DeepLIFT SHAP Concept Importance"
    if dataset_name:
        title = f"{dataset_name}: {title}"
    plt.title(title, fontsize=14)

    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot concept importance")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for figure")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for title")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    results = load_shap_results(exp_dir)

    if not results:
        print(f"No SHAP results found in {exp_dir}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = exp_dir / "concept_importance.pdf"

    plot_concept_importance(results, output_path, args.dataset)


if __name__ == "__main__":
    main()
