#!/usr/bin/env python3
"""Count how many measured CT bits differ from the reference at each voltage."""
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/ct_voltage/bit_flips.csv')
    args = parser.parse_args()
    source = json.loads((ROOT/'measurements/voltage/variants.json').read_text())
    reference = source['reference']
    assert len(reference) == 25 and set(reference) <= {'0', '1'}
    rows = []
    for group in sorted({v['group'] for v in source['variants']}, key=float):
        patterns = [v['pattern'] for v in source['variants'] if v['group'] == group]
        assert all(len(p) == 25 and set(p) <= {'0', '1'} for p in patterns)
        distances = [sum(a != b for a, b in zip(p, reference)) for p in patterns]
        rows.append(dict(voltage_v=float(group), supplied_entries=len(distances),
                         mean_bit_flips=statistics.mean(distances),
                         sample_standard_deviation=statistics.stdev(distances),
                         standard_deviation_over_sqrt_n=statistics.stdev(distances)/len(distances)**0.5))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
