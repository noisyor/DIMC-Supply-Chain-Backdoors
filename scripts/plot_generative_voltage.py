#!/usr/bin/env python3
"""Verify saved DiT image errors and plot success under measured CT variations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plot_discriminative_voltage import write_voltage_outputs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/generative_voltage"


def verify_patterns(name, protocol, source):
    """Check model identity and recompute each pattern's success from image errors."""
    folder = RESULTS / name
    report = json.loads((folder / "results.json").read_text())
    registry = json.loads((ROOT / "checkpoints/index.json").read_text())
    checkpoint = next(
        item for item in registry if item["id"] == report["checkpoint_id"]
    )
    assert report["model"] == checkpoint["matched_trigger"] == name
    assert report["checkpoint_sha256"] == checkpoint["sha256"]

    files = {
        ROOT / checkpoint["file"]: report["checkpoint_sha256"],
        ROOT / f"measurements/circuit/{name}.json": report["trigger_sha256"],
        folder / "per_sample_mse.npz": report["predictions_sha256"],
    }
    for path, expected_hash in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"File differs from the evaluated data: {path}")

    patterns = {entry["pattern"]: entry for entry in report["patterns"]}
    assert len(patterns) == len(report["patterns"])
    assert set(patterns) == {source["reference"]} | {
        entry["pattern"] for entry in source["variants"]
    }
    with np.load(folder / "per_sample_mse.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {entry["key"] for entry in patterns.values()}
        for entry in patterns.values():
            errors = saved[entry["key"]]
            assert errors.shape == (protocol["samples_per_pattern"],)
            assert np.isfinite(errors).all() and (errors >= 0).all()
            successes = int((errors < protocol["threshold"]).sum())
            assert successes == entry["successes"]
            assert successes / len(errors) == entry["asr"]
            assert float(errors.mean(dtype=np.float64)) == entry["mean_mse"]
    return patterns


def main():
    """Average ten voltage entries within each model, then combine five CT models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/generative_voltage"
    )
    args = parser.parse_args()
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    source_path = ROOT / "measurements/voltage/variants.json"
    assert (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        == protocol["inventory_sha256"]
    )
    assert protocol["precision"] == "FP32"
    assert protocol["threshold"] == 0.1 and protocol["samples_per_pattern"] == 1000
    source = json.loads(source_path.read_text())
    models = [verify_patterns(f"CT{i}", protocol, source) for i in range(1, 6)]

    rows = []
    for group in sorted({entry["group"] for entry in source["variants"]}, key=float):
        # Keep repeated entries: the measurement file supplies ten cases per voltage.
        patterns = [
            entry["pattern"] for entry in source["variants"] if entry["group"] == group
        ]
        distances = np.asarray(
            [
                sum(a != b for a, b in zip(pattern, source["reference"]))
                for pattern in patterns
            ]
        )
        model_means = np.asarray(
            [
                100 * np.mean([model[pattern]["asr"] for pattern in patterns])
                for model in models
            ]
        )
        row = {
            "voltage_v": float(group),
            "supplied_entries": len(patterns),
            "mean_bit_flips": float(distances.mean()),
            "bit_flips_sd_over_sqrt_n": float(
                distances.std(ddof=1) / np.sqrt(len(distances))
            ),
            "mean_asr_percent": float(model_means.mean()),
            "across_chip_asr_sd_percentage_points": float(model_means.std(ddof=1)),
        }
        row.update(
            {
                f"chip{i}_mean_asr_percent": float(value)
                for i, value in enumerate(model_means)
            }
        )
        rows.append(row)

    write_voltage_outputs(rows, args.output_dir, "generative_voltage")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
