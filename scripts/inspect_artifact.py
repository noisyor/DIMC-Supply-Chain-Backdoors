#!/usr/bin/env python3
"""Check file contents against saved hashes and summarize the trigger patterns."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    """Check trigger shapes and return pattern statistics and pairwise CT distances."""
    index = json.loads((ROOT / "triggers/index.json").read_text())
    rows = []
    bits = []

    # Validate the 5-by-5 mask and summarize only the values inside that region.
    for item in index:
        trigger_config = json.loads((ROOT / item["file"]).read_text())
        trigger = trigger_config["trigger"]
        mask = trigger_config["mask"][0]
        assert len(trigger) == 3 and all(
            len(c) == 32 and all(len(r) == 32 for r in c) for c in trigger
        )
        mask_positions = [(y, x) for y in range(32) for x in range(32) if mask[y][x]]
        assert all(mask[y][x] in (0, 1) for y in range(32) for x in range(32))
        first_row = min(y for y, x in mask_positions)
        first_column = min(x for y, x in mask_positions)
        assert mask_positions == [
            (y, x)
            for y in range(first_row, first_row + 5)
            for x in range(first_column, first_column + 5)
        ]
        trigger_values = [
            trigger[c][y][x]
            for c in range(3)
            for y in range(32)
            for x in range(32)
            if mask[y][x]
        ]
        assert all(math.isfinite(v) for v in trigger_values)
        row = {
            "id": item["id"],
            "family": item["family"],
            "mean": statistics.mean(trigger_values),
            "variance_population": statistics.pvariance(trigger_values),
            "spatial_positions": 25,
            "patch_origin_zero_based": [first_row, first_column],
            "paper_top_left_condition_satisfied": (first_row, first_column) == (0, 0),
        }
        if item["family"] == "circuit":
            assert all(v in (-1, 1) for v in trigger_values)
            assert trigger[0] == trigger[1] == trigger[2]
            b = [int(trigger[0][y][x] > 0) for y in range(5) for x in range(5)]
            row["hamming_weight"] = sum(b)
            row["paper_weight_constraint_satisfied"] = 8 <= sum(b) <= 17
            bits.append((item["id"], b))
        rows.append(row)
    assert len(set(tuple(b) for _, b in bits)) == 5

    # Count differing bits between each pair of measured CT patterns.
    matrix = [[sum(x != y for x, y in zip(a, b)) for _, b in bits] for _, a in bits]
    with (ROOT / "results/historical_training/losses.csv").open() as f:
        curve = list(csv.DictReader(f))

    # Check that each historical DiT log has ordered, non-repeated training steps.
    for ident in {r["trigger"] for r in curve}:
        steps = [int(r["step"]) for r in curve if r["trigger"] == ident]
        assert steps == sorted(set(steps)), ident
    return {
        "triggers": rows,
        "ct_pairwise_hamming_distance": matrix,
        "ct_order": [i for i, _ in bits],
        "historical_loss_rows": len(curve),
        "warning": "Each distance counts differing bits between two trigger patterns. It does not measure attack success or repeatability of chip measurements.",
    }


def verify():
    """Check every manifest entry and return the number of verified files."""
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    issues = []
    for row in manifest:
        artifact_path = ROOT / row["file"]
        if (
            not artifact_path.is_file()
            or hashlib.sha256(artifact_path.read_bytes()).hexdigest() != row["sha256"]
        ):
            issues.append(row["file"])
    if issues:
        raise SystemExit("Integrity check failed: " + ", ".join(issues))
    return len(manifest)


def main():
    """Write a trigger summary after the optional file-hash check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/inspection")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    count = None if args.skip_hashes else verify()
    summary = summarize()
    summary["verified_files"] = count
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trigger_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Verified files: {count}; triggers: {len(summary['triggers'])}; historical loss records: {summary['historical_loss_rows']}"
    )
    print(
        "AT2–AT5 patches start at row 3, column 4; the first row and column are numbered 0."
    )
    print(
        "The numbers of 1 bits in CT1–CT5 are 14, 2, 13, 11, and 14. See docs/SCOPE.md for the trigger labels."
    )


if __name__ == "__main__":
    main()
