# DIMC behavioral design and recorded patterns

`rtl/simulated_DIMC.sv` defines the `dimc_bank_row_acc_32x32_wReg` behavioral bank. It models SRAM weight storage, bit-serial signed activation accumulation, two weight-assembly modes, and parameter-dependent output packing. `rtl/simulated_DIMC_tb.sv` is the original integration testbench. These are behavioral simulation sources; complete fabricated-chip implementation and physical-design files are not included.

## Build and simulate

Install Python 3 and Verilator, with a C++ compiler and Make available. From the repository root, run:

```bash
python3 scripts/verify_dimc_rtl.py --lane-width 32 --output outputs/rtl32
python3 scripts/verify_dimc_rtl.py --lane-width 21 --output outputs/rtl21
python3 scripts/verify_dimc_rtl.py --lane-width 34 --output outputs/rtl34
```

The verifier generates an explicitly identified parameter fixture and stimulus harness, compiles the supplied behavioral module, and compares both weight modes with integer reference results. The released receipts cover 178 cases for 32-bit lanes and 66 each for 21- and 34-bit lanes. [The arithmetic guide](../docs/RTL_QUANTIZATION.md) specifies the operations, test scope, and runnable DiT Linear profile.

The original integration testbench separately references `microarch_parameters` and `../../data_weight.txt`, whose original versions are absent. `python3 scripts/check_rtl.py` reports these gaps and exits 2. The generated verification fixtures permit the commands above to run but do not recover the original full-chip integration or establish silicon equivalence.

## Measurement and reference files

| File | Meaning, units, and provenance |
|---|---|
| `../triggers/circuit/CT1.json` through `CT5.json` | Recorded SRAM-derived trigger patterns from source chip indices 0–4. Each JSON preserves a 3×32×32 trigger array, a spatial mask, and target image. The 25 spatial CT bits use −1/+1 encoding, repeated across RGB; mask values are 0/1. Image and trigger values are dimensionless model-input values. |
| `../triggers/relative_variants.json` | One 25-bit reference and ten supplied binary patterns per voltage group from 0.50 to 0.55 V. Groups are voltages in volts; each binary character is a spatial bit. The file does not identify acquisition frequency or map entries to individual chips versus repeated acquisitions. |
| `measurements/ct_voltage_bit_flips.csv` | Six computed rows from the supplied voltage patterns. `voltage_v`: volts; `supplied_entries`: count (10 per voltage); `mean_bit_flips`: mean Hamming distance to the reference in bits out of 25; `sample_standard_deviation`: sample SD in bits; `standard_deviation_over_sqrt_n`: sample SD divided by √10, also in bits. These statistics do not establish independent sampling. |
| `measurements/ct_voltage_plot.png` | Preserved source illustration. Some points and the attack-success curve lack underlying numerical records in this package; the image is not a numerical dataset for those values. |
| `measurements/legacy_mvm_outputs.csv` | 160 numerical reference rows. `TEST` and `COL`: categorical test/column IDs. `INT`, `BIN`, and `HEX`: decimal, binary, and hexadecimal representations of integer MVM outputs, with no physical unit or documented conversion scale. Acquisition conditions and linkage to other records are unverified. |
| `measurements/legacy_fingerprint.txt` | Legacy fingerprint symbols; X marks unknown/excluded positions. Acquisition conditions are unrecorded. This is not a complete startup measurement bank. |
| `analysis/weight_semantics.py` | Small numerical example of weight interpretations. |
| `analysis/legacy_sampling.py` | Archival model-quantization analysis with historical dependencies and defaults. The supported integer-Linear profile is documented separately. |

Recompute the voltage CSV without ML dependencies:

```bash
python3 scripts/summarize_ct_voltage.py --output outputs/ct_voltage/bit_flips.csv
```

See [CT voltage measurements](../docs/CT_VOLTAGE.md) for the distinction between measured bit patterns and software classification responses. Original contributions are covered by the root [MIT license](../LICENSE), subject to the [third-party terms](../THIRD_PARTY_NOTICES.md).
