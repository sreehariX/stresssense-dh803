"""Draw the aggregate results figure from a completed experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plot(summary_path, output):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = {
        "always_rest": "Always rest",
        "acc_only": "Movement only",
        "logistic": "Logistic regression",
        "svm_rbf": "RBF support vector machine",
        "random_forest": "Random forest",
        "gradient_boosting": "Gradient boosted trees",
        "gaussian_nb": "Gaussian naive Bayes",
    }
    models = [model for model in labels if model in summary["models"]]
    rows = np.arange(len(models))
    figure, axes = plt.subplots(figsize=(9.1, 1.5 + 0.7 * len(models)), layout="constrained")
    comparisons = [
        ("wesad_loso", "WESAD held out participants", -0.12, "o", "black"),
        ("physionet_transfer", "PhysioNet transfer", 0.12, "s", "#606060"),
    ]
    for study, label, offset, marker, ink in comparisons:
        records = [summary["models"][model][study] for model in models]
        means = np.array([row["mean"]["balanced_accuracy"] for row in records])
        intervals = np.array([row["bootstrap_95"]["balanced_accuracy"] for row in records])
        xerr = np.maximum(0, np.stack((means - intervals[:, 0], intervals[:, 1] - means)))
        axes.errorbar(means, rows + offset, xerr=xerr, fmt=marker, color=ink,
                      capsize=3, markersize=5, linewidth=1, label=label)
    axes.axvline(0.5, color="#999999", linestyle=":", linewidth=1)
    axes.set_xlim(0.25, 1.01)
    axes.set_yticks(rows, [labels[model] for model in models])
    axes.invert_yaxis()
    axes.set_xlabel("Mean participant balanced accuracy")
    axes.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("run/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("run/model_comparison.png"))
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
