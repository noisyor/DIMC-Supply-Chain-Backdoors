# Chip measurements and trigger patterns

This directory contains the chip-related data used to form architecture triggers (ATs), circuit triggers (CTs), and voltage-dependent CT variations. The [behavioral DIMC model](../hardware/README.md) and [software experiment results](../results/README.md) are separate.

| Directory | Contents |
|---|---|
| `architecture/` | Five saved AT configurations: `AT1.json` through `AT5.json`. |
| `circuit/` | SRAM power-up trigger patterns from chips 0–4, stored as `CT1.json` through `CT5.json`. |
| `voltage/` | CT patterns measured at different voltages, the number of changed bits, and the original plot. |
| `reference/` | Earlier MVM output and fingerprint records whose measurement settings are not fully recorded. |

## AT and CT formats

Each AT/CT JSON stores a `trigger` with three color channels and 32×32 positions per channel. A `mask` value of 1 replaces that input position with the trigger value; 0 leaves it unchanged. The target image is the output the poisoned DiT is trained to generate. These values have no physical units.

Each CT has 25 bits, encoded as −1 for 0 and +1 for 1, with the same pattern in all three color channels. AT1 and CT1–CT5 patches start at row 0, column 0; AT2–AT5 start at row 3, column 4. Row and column numbering begins at 0.

| AT | Configuration name |
|---|---|
| AT1 | weight_config |
| AT2 | all(−1) |
| AT3 | 24(−1) |
| AT4 | 24(+1) |
| AT5 | all(+1) |

The table lists configuration names; the JSON files contain the exact trigger values used in the experiments. Complete raw measurement logs and instrument settings are not included. The [trigger index](../configs/index.json) identifies each pattern. The `matched_trigger` field in the [DiT model list](../checkpoints/index.json) and [VGG model list](../checkpoints/classifier/index.json) tells you which trigger each model was trained to respond to.

## CT voltage variations

`voltage/variants.json` contains one 25-bit reference and ten recorded patterns at each voltage from 0.50 to 0.55 V. Each character represents one spatial bit. `voltage/bit_flips.csv` counts the bit positions that differ from that reference (Hamming distance):

| Column | Unit and meaning |
|---|---|
| `voltage_v` | Supply voltage, V. |
| `supplied_entries` | Number of recorded patterns; 10 per voltage. |
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

`reference/legacy_mvm_outputs.csv` contains 160 rows. `TEST` and `COL` identify tests and columns; `INT`, `BIN`, and `HEX` represent integer MVM outputs in decimal, binary, and hexadecimal. The records do not give a conversion from these integers to physical units or compare them with outputs from the behavioral model. `reference/legacy_fingerprint.txt` contains fingerprint symbols, with X marking unknown/excluded positions. The measurement settings were not recorded.

Original contributions use [MIT](../LICENSE), subject to [third-party terms](../THIRD_PARTY_NOTICES.md).
