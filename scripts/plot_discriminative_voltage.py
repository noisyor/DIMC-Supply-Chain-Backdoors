#!/usr/bin/env python3
"""Plot classifier success under measured CT voltage perturbations."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluate_classifier import verify_saved_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Verify voltage-condition predictions, aggregate across CT models, and plot the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/discriminative_voltage"
    )
    args = parser.parse_args()
    source = json.loads((ROOT / "measurements/voltage/variants.json").read_text())
    groups = sorted({v["group"] for v in source["variants"]}, key=float)

    # Average the ten supplied entries within each model before averaging across models.
    chip_means = []
    for chip in range(1, 6):
        name = f"CT{chip}"
        folder = ROOT / "results/software_campaign/classifier" / name
        report = json.loads((folder / "metrics.json").read_text())
        entries = verify_saved_model(folder, name)["relative"]
        assert len(entries) == len(source["variants"])
        with np.load(folder / f"{name}.npz") as predictions:
            eligible = predictions["labels"] != report["target"]
            assert eligible.sum() == 9000
            for entry, variant in zip(entries, source["variants"]):
                for key in ("group", "index_in_group", "pattern", "xor_from_reference"):
                    assert entry[key] == variant[key], (name, key)
                rate = np.mean(
                    predictions[entry["prediction_key"]][eligible] == report["target"]
                )
                assert abs(rate - entry["asr"]) < 1e-12
        chip_means.append(
            [
                100 * np.mean([e["asr"] for e in entries if e["group"] == g])
                for g in groups
            ]
        )
    chip_means = np.asarray(chip_means)

    # Bit-flip bars use SD/sqrt(10); ASR bars use sample SD across the five model means.
    rows = []
    for voltage_index, group in enumerate(groups):
        patterns = [v["pattern"] for v in source["variants"] if v["group"] == group]
        distances = np.asarray(
            [sum(a != b for a, b in zip(p, source["reference"])) for p in patterns]
        )
        row = dict(
            voltage_v=float(group),
            supplied_entries=len(patterns),
            mean_bit_flips=float(distances.mean()),
            bit_flips_sd_over_sqrt_n=float(
                distances.std(ddof=1) / np.sqrt(len(distances))
            ),
            mean_asr_percent=float(chip_means[:, voltage_index].mean()),
            across_chip_asr_sd_percentage_points=float(
                chip_means[:, voltage_index].std(ddof=1)
            ),
        )
        row.update(
            {
                f"chip{chip_index}_mean_asr_percent": float(
                    chip_means[chip_index, voltage_index]
                )
                for chip_index in range(5)
            }
        )
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "discriminative_voltage.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Plot measured bit changes and software attack success on separate vertical axes.
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, bit_axis = plt.subplots(figsize=(9, 3.5))
    asr_axis = bit_axis.twinx()
    voltages = [r["voltage_v"] for r in rows]
    bit_flip_curve = bit_axis.errorbar(
        voltages,
        [r["mean_bit_flips"] for r in rows],
        yerr=[r["bit_flips_sd_over_sqrt_n"] for r in rows],
        color="blue",
        marker="o",
        linewidth=1.6,
        markersize=7,
        capsize=3.5,
        label="Bit flips",
    )
    success_curve = asr_axis.errorbar(
        voltages,
        [r["mean_asr_percent"] for r in rows],
        yerr=[r["across_chip_asr_sd_percentage_points"] for r in rows],
        color="#e52329",
        marker="d",
        linestyle="--",
        linewidth=1.6,
        markersize=7,
        capsize=3.5,
        label="Success rate",
    )
    bit_axis.set(
        xlim=(0.495, 0.555), ylim=(-0.8, 16.5), xticks=voltages, yticks=[0, 5, 10, 15]
    )
    bit_axis.set_xticklabels([f"{v:.2f}" for v in voltages])
    asr_axis.set(ylim=(-5, 105), yticks=[0, 25, 50, 75, 100])
    bit_axis.set_xlabel("Voltage (V)", fontsize=20)
    bit_axis.set_ylabel("Bit flips of CTs", color="#1f77b4", fontsize=20)
    asr_axis.set_ylabel("Attack success rate (%)", color="#e52329", fontsize=15)
    bit_axis.tick_params(axis="y", labelcolor="#1f77b4")
    asr_axis.tick_params(axis="y", labelcolor="#e52329")
    for axis in (bit_axis, asr_axis):
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontweight("bold")
    figure.legend(
        [bit_flip_curve, success_curve],
        ["Bit flips", "Success rate"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        prop={"weight": "bold", "size": 16},
        frameon=False,
    )
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"discriminative_voltage.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
