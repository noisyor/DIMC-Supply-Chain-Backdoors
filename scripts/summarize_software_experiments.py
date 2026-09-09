#!/usr/bin/env python3
"""Verify software experiment records and generate compact CSV tables."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from evaluate_classifier import verify_saved_model

MODELS = (
    ["Clean"]
    + [f"{f}{i}" for f in ["AT", "CT"] for i in range(1, 6)]
    + ["legacy_white"]
)
MODES = ["FP32", "W8A32", "W8A8_clean", "W8A8_mixed"]
TRIGGERS = [f"{f}{i}" for f in ["AT", "CT"] for i in range(1, 6)] + ["legacy_white"]


def write_csv(path, rows):
    """Write the supplied rows with a header and stable column order."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Verify the complete software suite and rebuild its four summary tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = []
    matrix = []
    flips = []
    classifiers = []
    missing = []

    # Check every DiT model and precision setting before producing summary rows.
    for name in MODELS:
        for mode in MODES:
            folder = args.root / "evaluation" / name / mode
            needed = ["matrix.json", "quality.json", "per_sample_mse.npz"] + (
                ["ct_flips.json"] if name.startswith("CT") else []
            )
            if not all((folder / n).exists() for n in needed):
                missing.append(f"DiT/{name}/{mode}")
                continue
            matrix_record = json.loads((folder / "matrix.json").read_text())
            quality_record = json.loads((folder / "quality.json").read_text())
            saved_arrays = np.load(folder / "per_sample_mse.npz", allow_pickle=False)
            assert (
                matrix_record["samples"] == 1000 and quality_record["samples"] == 50000
            )
            assert (
                matrix_record["seed"] == 4042
                and quality_record["seed"] == 3042
                and matrix_record["threshold"] == 0.1
            )
            assert (
                matrix_record["quantization"]["mode"] == mode
                and quality_record["precision"].split(";")[0] == mode
            )
            assert (
                matrix_record["checkpoint_sha256"]
                == quality_record["checkpoint_sha256"]
            )

            # Matrix and quality runs must use matching quantization and calibration settings.
            if mode != "FP32":
                matrix_quantization = matrix_record["quantization"]
                quality_quantization = quality_record["quantization"]
                for key in [
                    "mode",
                    "weight_range",
                    "weight_scale",
                    "activation_scale",
                    "rounding",
                    "calibration_seed",
                    "calibration_samples",
                    "layers",
                    "arithmetic",
                    "silicon_equivalence",
                ]:
                    assert matrix_quantization[key] == quality_quantization[key]
                assert set(matrix_quantization["calibration_activation_max"]) == set(
                    quality_quantization["calibration_activation_max"]
                )
                for key in matrix_quantization["calibration_activation_max"]:
                    assert np.isclose(
                        matrix_quantization["calibration_activation_max"][key],
                        quality_quantization["calibration_activation_max"][key],
                        rtol=1e-6,
                        atol=1e-8,
                    )
            assert set(matrix_record["matrix"]) == set(TRIGGERS)
            assert matrix_record["paired"] == (
                name if name != "Clean" else "legacy_white"
            )
            assert np.isfinite(
                [
                    quality_record["metrics"]["frechet_inception_distance"],
                    quality_record["metrics"]["inception_score_mean"],
                ]
            ).all()

            def verify(condition, key):
                """Recompute one DiT condition from its saved per-sample errors."""
                mse_values = saved_arrays[key].astype(np.float64)
                assert mse_values.shape == (1000,) and np.isfinite(mse_values).all()
                assert int((mse_values < 0.1).sum()) == condition["successes"]
                assert (
                    abs(100 * (mse_values < 0.1).mean() - condition["bsr_percent"])
                    < 1e-8
                )
                assert abs(mse_values.mean() - condition["mean_mse"]) < 1e-8

            verify(matrix_record["clean_target_rate"], "clean")
            for trigger, rec in matrix_record["matrix"].items():
                verify(rec, trigger)
                matrix.append(
                    {"model": name, "precision": mode, "trigger": trigger, **rec}
                )
            paired = matrix_record["paired"]
            summary.append(
                {
                    "model": name,
                    "precision": mode,
                    "matched_bsr_percent": (
                        None
                        if name == "Clean"
                        else matrix_record["matrix"][paired]["bsr_percent"]
                    ),
                    "clean_target_rate_percent": matrix_record["clean_target_rate"][
                        "bsr_percent"
                    ],
                    "fid": quality_record["metrics"]["frechet_inception_distance"],
                    "inception_score": quality_record["metrics"][
                        "inception_score_mean"
                    ],
                }
            )

            # Reconstruct the fixed-seed masks and check each saved perturbation result.
            if name.startswith("CT"):
                evaluation_record = json.loads((folder / "ct_flips.json").read_text())
                assert len(evaluation_record["rows"]) == 146
                assert (
                    evaluation_record["seed"] == 6042
                    and evaluation_record["samples_per_pattern"] == 1000
                    and evaluation_record["shared_noise_seed"] == 4042
                )
                rng = np.random.default_rng(6042)
                for count in range(8):
                    rows = [x for x in evaluation_record["rows"] if x["flips"] == count]
                    assert len(rows) == (1 if count == 0 else 25 if count == 1 else 20)
                    patterns = (
                        [[]]
                        if count == 0
                        else (
                            [[i] for i in range(25)]
                            if count == 1
                            else [
                                sorted(rng.choice(25, count, replace=False).tolist())
                                for _ in range(20)
                            ]
                        )
                    )
                    for i, (rec, pattern) in enumerate(zip(rows, patterns)):
                        assert rec["pattern_index"] == i and rec["positions"] == pattern
                        verify(rec, f"flip_{count}_{i}")
                    flips.append(
                        {
                            "model": name,
                            "precision": mode,
                            "flips": count,
                            "patterns": len(rows),
                            "mean_bsr_percent": float(
                                np.mean([x["bsr_percent"] for x in rows])
                            ),
                            "min_bsr_percent": min(x["bsr_percent"] for x in rows),
                            "max_bsr_percent": max(x["bsr_percent"] for x in rows),
                        }
                    )

    # Check classifier predictions against the released weights and updated training variant.
    for name in ["Clean", "White"] + [
        f"{f}{i}" for f in ["AT", "CT"] for i in range(1, 6)
    ]:
        folder = args.root / "classifier" / name
        if (
            not (folder / "metrics.json").exists()
            or not (folder / (name + ".npz")).exists()
        ):
            missing.append("classifier/" + name)
            continue
        evaluation_record = verify_saved_model(folder, name)
        saved_arrays = np.load(folder / (name + ".npz"))
        labels = saved_arrays["labels"]
        assert len(labels) == 10000
        assert (
            abs(
                (saved_arrays["clean"] == labels).mean()
                - evaluation_record["clean_accuracy"]
            )
            < 1e-12
        )
        keep = labels != 2
        assert keep.sum() == 9000
        for trigger, rec in evaluation_record["cross"].items():
            predictions = saved_arrays["cross_" + trigger]
            assert abs((predictions[keep] == 2).mean() - rec["asr"]) < 1e-12
            assert (
                abs((predictions == labels).mean() - rec["triggered_accuracy"]) < 1e-12
            )
        for rec in evaluation_record["random"] + evaluation_record["relative"]:
            predictions = saved_arrays[rec["prediction_key"]]
            assert abs((predictions[keep] == 2).mean() - rec["asr"]) < 1e-12
            assert (
                abs((predictions == labels).mean() - rec["triggered_accuracy"]) < 1e-12
            )
        assert set(evaluation_record["cross"]) == set(TRIGGERS[:-1] + ["White"])
        if name.startswith("CT"):
            assert (
                len(evaluation_record["random"]) == 146
                and len(evaluation_record["relative"]) == 60
            )
        classifiers.append(
            {
                "model": name,
                "training_variant": evaluation_record["training_variant"],
                "clean_accuracy_percent": 100 * evaluation_record["clean_accuracy"],
                "matched_asr_percent": (
                    None
                    if name == "Clean"
                    else 100 * evaluation_record["cross"][name]["asr"]
                ),
                "max_nonmatching_asr_percent": (
                    None
                    if name == "Clean"
                    else 100
                    * max(
                        v["asr"]
                        for k, v in evaluation_record["cross"].items()
                        if k != name
                    )
                ),
                "cross_trigger_conditions": len(evaluation_record["cross"]),
                "random_conditions": len(evaluation_record["random"]),
                "relative_conditions": len(evaluation_record["relative"]),
            }
        )

    # A missing model or condition makes the suite incomplete, even if other results pass.
    report = {
        "complete": not missing,
        "dit_quality_rows": len(summary),
        "dit_matrix_rows": len(matrix),
        "ct_perturbation_rows": len(flips),
        "classifier_models": len(classifiers),
        "missing": missing,
    }
    output_dir = args.root / "summary"
    output_dir.mkdir(exist_ok=True)
    for name, rows in [
        ("dit_quality", summary),
        ("dit_matrix", matrix),
        ("ct_perturbations", flips),
        ("classifiers", classifiers),
    ]:
        if rows:
            write_csv(output_dir / (name + ".csv"), rows)
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if missing and not args.allow_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
