#!/usr/bin/env python3
"""Plot AT and CT classifier comparisons from verified, current predictions."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluate_classifier import verify_saved_model

ROOT = Path(__file__).resolve().parents[1]


def plot_family(family, output_dir):
    """Save a five-model comparison using the public model and trigger labels."""
    names = [f"{family}{i}" for i in range(1, 6)]
    reports = [
        verify_saved_model(ROOT / "results/software_campaign/classifier" / name, name)
        for name in names
    ]
    assert all(
        report["training_variant"] == "nonmatching_trigger_loss_v1"
        for report in reports
    )
    rows = [
        {
            "model": name,
            "trigger": trigger,
            "asr_percent": 100 * report["cross"][trigger]["asr"],
            "clean_accuracy_percent": 100 * report["clean_accuracy"],
        }
        for name, report in zip(names, reports)
        for trigger in names
    ]
    stem = f"{family.lower()}_classifier_comparison"
    with (output_dir / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    values = np.asarray([row["asr_percent"] for row in rows]).reshape(5, 5)
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    heatmap = axis.imshow(values, cmap="Blues", vmin=0, vmax=100)
    axis.set_xticks(range(5))
    axis.set_xticklabels(names)
    axis.set_yticks(range(5))
    axis.set_yticklabels(
        [
            f"{name} ({100 * report['clean_accuracy']:.2f}%)"
            for name, report in zip(names, reports)
        ]
    )
    axis.set_xlabel("Input trigger")
    axis.set_ylabel("Trained model (clean accuracy)")
    for row in range(5):
        for column in range(5):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 60 else "black",
                fontsize=12,
            )
    colorbar = figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Attack success rate (%)")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Verified and plotted {family}: 25 model–trigger comparisons.")


def main():
    """Generate separate figures for architecture and circuit triggers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/classifier_comparisons"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    for family in ("AT", "CT"):
        plot_family(family, args.output_dir)


if __name__ == "__main__":
    main()
