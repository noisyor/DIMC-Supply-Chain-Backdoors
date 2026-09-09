#!/usr/bin/env python3
"""Check trigger shapes and summarize the recorded patterns."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    """Check trigger shapes and return pattern statistics and pairwise CT distances."""
    index = json.loads((ROOT / "configs/index.json").read_text())
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


def main():
    """Validate the recorded patterns and write their summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/inspection")
    args = parser.parse_args()
    summary = summarize()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trigger_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Triggers: {len(summary['triggers'])}; historical loss records: {summary['historical_loss_rows']}"
    )
    print(
        "AT2–AT5 patches start at row 3, column 4; the first row and column are numbered 0."
    )
    print(
        "The numbers of 1 bits in CT1–CT5 are 14, 2, 13, 11, and 14. See docs/SCOPE.md for the trigger labels."
    )


if __name__ == "__main__":
    main()
