# Chip measurements and trigger patterns

This directory contains the chip-related data used to form architecture triggers (ATs), circuit triggers (CTs), and voltage-dependent CT variations. The [behavioral DIMC model](../hardware/README.md) and [software experiment results](../results/README.md) are separate.

| Directory | Contents |
|---|---|
| `architecture/` | Five saved AT configurations: `AT1.json` through `AT5.json`. |
| `circuit/` | Five measured SRAM-derived CT patterns: `CT1.json` through `CT5.json`, measured from chips 0–4. |
| `voltage/` | Recorded CT variants, their computed bit-flip summary, and the supplied source plot. |
| `reference/` | Earlier MVM output and fingerprint records whose measurement settings are not fully recorded. |

## AT and CT formats

Each AT/CT JSON contains a 3×32×32 `trigger` array, a spatial `mask`, and a target image for software evaluation. Values are dimensionless model-input values; masks use 0/1. A CT contains 25 spatial bits, encoded as −1/+1 and repeated across RGB. Mask origins are zero-based: AT1 and CT1–CT5 use (0,0); AT2–AT5 use (3,4).

| AT | Configuration |
|---|---|
| AT1 | weight_config |
| AT2 | all(−1) |
| AT3 | 24(−1) |
| AT4 | 24(+1) |
| AT5 | all(+1) |

These files store the trigger values used in the experiments. Complete raw measurement logs and instrument settings are not included. The [trigger index](../triggers/index.json) identifies each pattern. The `matched_trigger` field in the [DiT model list](../checkpoints/index.json) and [VGG model list](../checkpoints/classifier/index.json) tells you which trigger each model was trained to respond to.

## CT voltage variations

`voltage/variants.json` contains one 25-bit reference and ten supplied patterns at each voltage from 0.50 to 0.55 V. Each character represents one spatial bit. `voltage/bit_flips.csv` summarizes Hamming distance from that reference:

| Column | Unit and meaning |
|---|---|
| `voltage_v` | Supply voltage, V. |
| `supplied_entries` | Number of supplied patterns; 10 per voltage. |
| `mean_bit_flips` | Mean differing bits out of 25. |
| `sample_standard_deviation` | Sample standard deviation, bits. |
| `standard_deviation_over_sqrt_n` | Sample standard deviation divided by √10, bits. |

Repeated patterns are counted each time they appear. The records do not state how often measurements were taken or whether the entries came from different chips or repeated measurements of one chip. We therefore cannot assume that the entries are independent measurements. The source plot includes additional points without numerical records, which are outside the released numerical dataset.

From the repository root, recompute the CSV with Python 3:

```bash
python scripts/summarize_ct_voltage.py --output outputs/voltage/bit_flips.csv
```

[Voltage evaluation](../docs/CT_VOLTAGE.md) explains how software experiments apply these measured differences to each model's own CT.

## Earlier reference records

`reference/legacy_mvm_outputs.csv` contains 160 rows. `TEST` and `COL` identify tests and columns; `INT`, `BIN`, and `HEX` represent integer MVM outputs in decimal, binary, and hexadecimal. No physical conversion scale or matched behavioral-model comparison is recorded. `reference/legacy_fingerprint.txt` contains fingerprint symbols, with X marking unknown/excluded positions. The measurement settings were not recorded.

Original contributions use [MIT](../LICENSE), subject to [third-party terms](../THIRD_PARTY_NOTICES.md).
