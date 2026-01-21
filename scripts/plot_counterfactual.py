"""
Plot counterfactual intervention accuracy (Figure 7 from paper).

This script visualizes how well models respond to counterfactual concept interventions
on the AA2 dataset (brown bear -> polar bear by setting 'white' concept).

Usage:
    python scripts/plot_counterfactual.py --exp-dir <EXPERIMENT_DIR>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


METHOD_NAMES = {
    "bottleneck": "CBM",
    "latent_residual": "Latent Residual",
    "decorrelated_residual": "Decorrelated Residual",
    "iter_norm": "Iterative Norm",
    "eye": "EYE",
    "adversarial_decorrelation": "Adversarial Decorrelation",
    "mi_residual": "MI Residual",
    "cem": "CEM",
}

METHOD_ORDER = [
    "bottleneck",
    "latent_residual",
    "decorrelated_residual",
    "iter_norm",
    "eye",
    "adversarial_decorrelation",
    "mi_residual",
]


def load_counterfactual_results(exp_dir: Path):
    """Load counterfactual results from experiment directory."""
    results = {}
    plots_dir = exp_dir / "plots"

    if not plots_dir.exists():
        # Try eval directory
        plots_dir = exp_dir / "eval"

    if not plots_dir.exists():
        raise FileNotFoundError(f"Plots/eval directory not found in: {exp_dir}")

    # Look for CSV files with counterfactual results
    for method in METHOD_ORDER:
        pattern = f"*{method}*counterfactual*.csv"
        files = list(plots_dir.glob(pattern))
        if files:
            results[method] = pd.read_csv(files[0])

    return results


def plot_counterfactual_accuracy(
    results: dict,
    output_path: Path,
):
    """Plot counterfactual intervention accuracy vs residual dimension."""
    plt.figure(figsize=(8, 5))

    for method, df in results.items():
        if method not in METHOD_NAMES:
            continue

        # Look for accuracy column
        acc_col = None
        for col in df.columns:
            if "acc" in col.lower() or "counterfactual" in col.lower():
                if "std" not in col.lower():
                    acc_col = col
                    break

        if acc_col is None:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                acc_col = numeric_cols[0]

        if acc_col is None:
            continue

        x = np.arange(len(df))
        plt.plot(x, df[acc_col], '-o', label=METHOD_NAMES[method], markersize=8)

    # Format plot
    plt.xlabel("Residual Dimension", fontsize=12)
    plt.ylabel("Counterfactual Accuracy", fontsize=12)
    plt.title("Counterfactual Intervention Accuracy (AA2 Dataset)", fontsize=14)

    # Add reference line at 100%
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Response')

    plt.legend(loc="lower left", fontsize=10)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot counterfactual accuracy")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for figure")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    results = load_counterfactual_results(exp_dir)

    if not results:
        print(f"No counterfactual results found in {exp_dir}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = exp_dir / "counterfactual_accuracy.pdf"

    plot_counterfactual_accuracy(results, output_path)


if __name__ == "__main__":
    main()
