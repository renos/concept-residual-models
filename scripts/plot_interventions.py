"""
Plot intervention accuracy degradation figures (Figure 4 and Figure 5 from paper).

Usage:
    python scripts/plot_interventions.py --exp-dir <EXPERIMENT_DIR> --type concepts
    python scripts/plot_interventions.py --exp-dir <EXPERIMENT_DIR> --type residual
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


def load_results(exp_dir: Path, intervention_type: str):
    """Load intervention results from experiment directory."""
    results = {}
    plots_dir = exp_dir / "plots"

    if not plots_dir.exists():
        raise FileNotFoundError(f"Plots directory not found: {plots_dir}")

    # Look for CSV files with intervention results
    for method in METHOD_ORDER:
        if intervention_type == "concepts":
            pattern = f"*{method}*random*.csv"
        else:
            pattern = f"*{method}*residual*.csv"

        files = list(plots_dir.glob(pattern))
        if files:
            results[method] = pd.read_csv(files[0])

    return results


def plot_intervention_degradation(
    results: dict,
    intervention_type: str,
    output_path: Path,
    dataset_name: str = None,
):
    """Plot intervention accuracy degradation."""
    plt.figure(figsize=(8, 5))

    for method, df in results.items():
        if method not in METHOD_NAMES:
            continue

        # Calculate degradation
        if "Baseline Acc." in df.columns and "Random Concepts Acc." in df.columns:
            degradation = df["Baseline Acc."] - df["Random Concepts Acc."]
            std = np.sqrt(df.get("Baseline Std.", 0)**2 + df.get("Random Concepts Std.", 0)**2)
        elif "Baseline Acc." in df.columns and "Random Residual Acc." in df.columns:
            degradation = df["Baseline Acc."] - df["Random Residual Acc."]
            std = np.sqrt(df.get("Baseline Std.", 0)**2 + df.get("Random Residual Std.", 0)**2)
        else:
            continue

        x = np.arange(len(degradation))
        plt.errorbar(x, degradation, yerr=std, fmt='-o', label=METHOD_NAMES[method])

    # Format x-axis as residual dimensions
    plt.xlabel("Residual Dimension", fontsize=12)
    plt.ylabel("Accuracy Degradation", fontsize=12)

    title_type = "Concept" if intervention_type == "concepts" else "Residual"
    title = f"Accuracy Degradation from Random {title_type} Intervention"
    if dataset_name:
        title = f"{dataset_name}: {title}"
    plt.title(title, fontsize=14)

    plt.legend(loc="best", fontsize=10)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot intervention degradation")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--type", choices=["concepts", "residual"], required=True,
                        help="Type of intervention to plot")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for figure (default: auto-generated)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for title")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    results = load_results(exp_dir, args.type)

    if not results:
        print(f"No results found in {exp_dir}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = exp_dir / f"negative_{args.type}_intervention_degradation.pdf"

    plot_intervention_degradation(results, args.type, output_path, args.dataset)


if __name__ == "__main__":
    main()
