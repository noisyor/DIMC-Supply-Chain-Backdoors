#!/usr/bin/env python3
"""Create VGG and DiT comparison tables from verified released results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from evaluate_classifier import verify_saved_model

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/software_campaign"
FAMILIES = {family: [f"{family}{i}" for i in range(1, 6)] for family in ("AT", "CT")}
START = "<!-- BEGIN MODEL COMPARISONS -->"
END = "<!-- END MODEL COMPARISONS -->"


def write_csv(path, rows):
    """Keep full numeric precision in the downloadable data."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_dit(name, precision, checkpoint):
    """Verify image errors and read FID for the same model and precision."""
    folder = CAMPAIGN / "evaluation" / name / precision
    matrix = json.loads((folder / "matrix.json").read_text())
    quality = json.loads((folder / "quality.json").read_text())
    assert (
        matrix["checkpoint_sha256"]
        == quality["checkpoint_sha256"]
        == checkpoint["sha256"]
    )
    assert matrix["samples"] == 1000 and matrix["seed"] == 4042
    assert matrix["threshold"] == 0.1 and matrix["quantization"]["mode"] == precision
    assert matrix["paired"] == ("legacy_white" if name == "Clean" else name)
    assert quality["samples"] == 50000 and quality["seed"] == 3042
    assert quality["precision"].split(";")[0] == precision
    assert quality["reference"] == "CIFAR-10 train, 50000 images"

    # BSR and MSE are recomputed; FID comes from the saved 50,000-image evaluation.
    with np.load(folder / "per_sample_mse.npz", allow_pickle=False) as saved:
        for trigger, metrics in matrix["matrix"].items():
            errors = saved[trigger].astype(np.float64)
            assert errors.shape == (1000,) and np.isfinite(errors).all()
            assert (errors >= 0).all()
            successes = int((errors < 0.1).sum())
            assert successes == metrics["successes"]
            assert abs(100 * successes / len(errors) - metrics["bsr_percent"]) < 1e-8
            assert abs(errors.mean() - metrics["mean_mse"]) < 1e-8
    fid = quality["metrics"]["frechet_inception_distance"]
    assert np.isfinite(fid)
    return {"matrix": matrix["matrix"], "fid": fid}


def load_results():
    """Load current VGG predictions and FP32 and weight-only INT8 DiT records."""
    names = ["Clean"] + FAMILIES["AT"] + FAMILIES["CT"]
    classifiers = {
        name: verify_saved_model(CAMPAIGN / "classifier" / name, name) for name in names
    }
    assert all(
        classifiers[name]["training_variant"] == "nonmatching_trigger_loss_v1"
        for name in names
        if name != "Clean"
    )
    registry = {
        record["matched_trigger"] or "Clean": record
        for record in json.loads((ROOT / "checkpoints/index.json").read_text())
    }
    generative = {}
    for name in names:
        checkpoint = registry[name]
        actual_hash = hashlib.sha256(
            (ROOT / checkpoint["file"]).read_bytes()
        ).hexdigest()
        if actual_hash != checkpoint["sha256"]:
            raise ValueError(
                f"DiT checkpoint differs from the released weights: {name}"
            )
        generative[name] = {
            precision: read_dit(name, precision, checkpoint)
            for precision in ("FP32", "W8A32")
        }
    return classifiers, generative


def comparison_table(family, kind, classifiers, generative, output_dir):
    """Put BSR with clean accuracy or MSE in every model–trigger cell."""
    names = FAMILIES[family]
    rows, cells, rates = [], [], []
    for name in names:
        cell_row, rate_row = [name], []
        for trigger in names:
            if kind == "classifier":
                report = classifiers[name]
                rate = 100 * report["cross"][trigger]["asr"]
                secondary = 100 * report["clean_accuracy"]
                extra = {"clean_accuracy_percent": secondary}
                text = f"{rate:.2f} ({secondary:.2f})"
            else:
                metrics = generative[name]["FP32"]["matrix"][trigger]
                rate, secondary = metrics["bsr_percent"], metrics["mean_mse"]
                extra = {"precision": "FP32", "mean_mse": secondary}
                text = f"{rate:.1f} ({secondary:.4f})"
            rows.append(
                {"model": name, "trigger": trigger, "bsr_percent": rate, **extra}
            )
            cell_row.append(text)
            rate_row.append(rate)
        cells.append(cell_row)
        rates.append(rate_row)

    stem = f"{family.lower()}_{kind}_comparison"
    write_csv(output_dir / f"{stem}.csv", rows)
    figure, axis = plt.subplots(figsize=(10.5, 2.35))
    axis.axis("off")
    title = (
        "VGG-16: BSR % (clean accuracy %)"
        if kind == "classifier"
        else "DiT, FP32: BSR % (MSE)"
    )
    axis.set_title(
        f"{family} transfer — {title}", fontsize=16, fontweight="bold", pad=12
    )
    table = axis.table(
        cellText=cells,
        colLabels=["Model"] + names,
        cellLoc="center",
        colWidths=[0.075] + [0.185] * 5,
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    colors = LinearSegmentedColormap.from_list(
        "success", ["#ffa6a6", "#ffefa2", "#9bea76"]
    )
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.7)
        if row == 0 or column == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor(colors(rates[row - 1][column - 1] / 100))
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def performance_summaries(classifiers, generative, output_dir):
    """Average over five models for each AT/CT performance summary row."""
    vgg_rows, dit_rows = [], []
    for family, names in {"Clean": ["Clean"], **FAMILIES}.items():
        clean = family == "Clean"
        vgg_rows.append(
            {
                "model_group": family,
                "model_count": len(names),
                "clean_accuracy_percent": float(
                    np.mean([100 * classifiers[n]["clean_accuracy"] for n in names])
                ),
                "matched_bsr_percent": (
                    None
                    if clean
                    else float(
                        np.mean(
                            [100 * classifiers[n]["cross"][n]["asr"] for n in names]
                        )
                    )
                ),
            }
        )
        dit_rows.append(
            {
                "model_group": family,
                "model_count": len(names),
                "fp32_fid": float(
                    np.mean([generative[n]["FP32"]["fid"] for n in names])
                ),
                "w8a32_fid": float(
                    np.mean([generative[n]["W8A32"]["fid"] for n in names])
                ),
                "fp32_matched_mean_mse": (
                    None
                    if clean
                    else float(
                        np.mean(
                            [
                                generative[n]["FP32"]["matrix"][n]["mean_mse"]
                                for n in names
                            ]
                        )
                    )
                ),
                "fp32_matched_bsr_percent": (
                    None
                    if clean
                    else float(
                        np.mean(
                            [
                                generative[n]["FP32"]["matrix"][n]["bsr_percent"]
                                for n in names
                            ]
                        )
                    )
                ),
            }
        )
    write_csv(output_dir / "classifier_performance.csv", vgg_rows)
    write_csv(output_dir / "generative_performance.csv", dit_rows)
    return vgg_rows, dit_rows


def documentation(vgg_rows, dit_rows):
    """Generate the performance summaries and links to four comparison tables."""
    lines = [
        "### Performance summaries",
        "",
        "AT/CT rows average five models. Matched BSR uses each model's own trigger; N/A means no matched backdoor. FID and MSE are dimensionless.",
        "",
        "**Discriminative models (VGG-16, weight-only INT8)**",
        "",
        "| Model | Clean accuracy (%) | Matched BSR (%) |",
        "|---|---:|---:|",
    ]
    for row in vgg_rows:
        rate = row["matched_bsr_percent"]
        rate_text = "N/A" if rate is None else f"{rate:.2f}"
        lines.append(
            f"| {row['model_group']} | {row['clean_accuracy_percent']:.2f} | {rate_text} |"
        )
    lines += [
        "",
        "**Generative models (DiT)**",
        "",
        "| Model | FP32 FID | INT8 FID (W8A32) | Backdoor MSE (FP32) | Matched BSR (FP32, %) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in dit_rows:
        mse, rate = row["fp32_matched_mean_mse"], row["fp32_matched_bsr_percent"]
        mse_text = "N/A" if mse is None else f"{mse:.4f}"
        rate_text = "N/A" if rate is None else f"{rate:.2f}"
        lines.append(
            f"| {row['model_group']} | {row['fp32_fid']:.2f} | {row['w8a32_fid']:.2f} | {mse_text} | {rate_text} |"
        )
    lines += [
        "",
        "[VGG summary CSV](../results/model_comparisons/classifier_performance.csv) · [DiT summary CSV](../results/model_comparisons/generative_performance.csv)",
        "",
        "### AT and CT transfer tables",
        "",
        "Rows are trained models; columns are applied triggers. VGG cells show **BSR % (clean accuracy %)**; DiT cells show **BSR % (MSE)** in FP32. Green indicates higher BSR; red indicates lower BSR.",
        "",
    ]
    for family in FAMILIES:
        for kind, label in (("classifier", "VGG-16"), ("generative", "DiT")):
            stem = f"{family.lower()}_{kind}_comparison"
            path = "../results/model_comparisons/" + stem
            lines += [
                f"![{family} transfer for {label}]({path}.png)",
                "",
                f"[{family} {label} PDF]({path}.pdf) · [{family} {label} CSV]({path}.csv)",
                "",
            ]
    lines += [
        "Recreate the tables from saved results with NumPy and Matplotlib:",
        "",
        "```bash",
        "python scripts/plot_model_comparisons.py --output-dir outputs/model_comparisons",
        "```",
        "",
        "The script checks saved arrays and checkpoint identities; it does not rerun inference or FID.",
    ]
    return "\n".join(lines)


def main():
    """Write figures and data tables, and optionally refresh the guide."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/model_comparisons"
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help="Refresh the comparison section in docs/SOFTWARE_EXPERIMENTS.md.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
    classifiers, generative = load_results()
    for family in FAMILIES:
        for kind in ("classifier", "generative"):
            comparison_table(family, kind, classifiers, generative, args.output_dir)
    vgg_rows, dit_rows = performance_summaries(classifiers, generative, args.output_dir)
    if args.update_docs:
        path = ROOT / "docs/SOFTWARE_EXPERIMENTS.md"
        before, rest = path.read_text().split(START)
        _, after = rest.split(END)
        path.write_text(
            before
            + START
            + "\n\n"
            + documentation(vgg_rows, dit_rows)
            + "\n\n"
            + END
            + after
        )
    print(
        "Verified 11 VGG models and 11 DiT models; wrote four transfer tables and two performance summaries."
    )


if __name__ == "__main__":
    main()
