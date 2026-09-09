# Data guide

These records describe target-image generation, targeted classification, trigger sensitivity, and software quantization. CSV files contain actual recorded or computed values. JSON/JSONL files retain protocols and individual evaluation records; NPZ files retain arrays that can be opened with `numpy.load(path, allow_pickle=False)`.

## Main software datasets

All paths below are relative to this directory. CSV files have a header and use decimal points. Empty matched-rate fields for the clean model mean not applicable, not zero. Model, trigger, and precision columns are categorical identifiers. AT1–AT5 and CT1–CT5 select the configurations in `../triggers/index.json`; `legacy_white` is the DiT white-patch baseline and `White` is the classifier baseline.

| CSV | Rows | Meaning and column units |
|---|---:|---|
| `software_campaign/summary/dit_matrix.csv` | 528 | Twelve models × four precision settings × eleven triggers. `mean_mse`: mean squared error in dimensionless normalized image values; `successes`: count among 1,000 inputs; `bsr_percent`: percent with target MSE strictly below 0.1. `model`, `precision`, `trigger`: identifiers. |
| `software_campaign/summary/dit_quality.csv` | 48 | One row per model and precision. `matched_bsr_percent` and `clean_target_rate_percent`: percent of 1,000 inputs that generate the target with and without the paired trigger. `fid`: dimensionless Fréchet Inception Distance; `inception_score`: dimensionless mean Inception Score, both computed on 50,000 clean generations. |
| `software_campaign/summary/ct_perturbations.csv` | 160 | Five CT models × four precision settings × eight flip counts. `flips`: number of altered spatial bits out of 25, applied identically across RGB; `patterns`: number of tested masks. `mean_bsr_percent`, `min_bsr_percent`, `max_bsr_percent`: summary of per-mask BSR in percent, with 1,000 inputs per mask. |
| `software_campaign/summary/classifiers.csv` | 12 | One row per VGG model. `clean_accuracy_percent`: percent correct among 10,000 CIFAR-10 test images. `matched_asr_percent`: percent classified as bird (class 2) among 9,000 non-bird test images with the paired trigger. `cross_trigger_conditions`, `random_conditions`, `relative_conditions`: counts of evaluated conditions. |
| `discriminative_voltage/discriminative_voltage.csv` | 6 | Software classifier response to measured voltage-pattern differences. `voltage_v`: volts; `supplied_entries`: count; `mean_bit_flips` and `bit_flips_sd_over_sqrt_n`: bits out of 25. `mean_asr_percent` and `chip0_mean_asr_percent` through `chip4_mean_asr_percent`: percent targeted success, averaged over ten perturbations per classifier. `across_chip_asr_sd_percentage_points`: sample standard deviation of the five classifier means in percentage points. |

### How the values were generated

DiT uses one-step inference at `t=0`, balanced labels across ten classes, and noise seed 4042 for model–trigger comparisons. MSE compares the raw output with the normalized target, before image clipping. FID and Inception Score use seed 3042, rounded and clipped uint8 image pixels, and torch-fidelity 0.3.0; FID uses the 50,000 CIFAR-10 training images as its reference. FP32, W8A32, W8A8_clean, and W8A8_mixed are separate software arithmetic/calibration settings, defined in [the protocol guide](../docs/SOFTWARE_EXPERIMENTS.md). The quantization settings use floating-point operators after quantization and dequantization.

Controlled DiT bit-flip experiments use mask seed 6042: the unchanged trigger, all 25 single-bit masks, and 20 masks for each count from two through seven. Each mask uses the same 1,000-input noise bank. Classifier controlled masks use seed 20260907 and the full test set. Classifier accuracy and ASR values inside JSON records are fractions in [0,1]; the summary CSV converts them to percent.

The voltage classifier evaluation applies `base XOR (pattern XOR reference)` to each CT model's own trigger. It measures software response to recorded variations, not classifier operation on a chip at that voltage. Repeated patterns are counted each time they appear. Error-bar definitions and the available measurement details are in [the voltage guide](../docs/CT_VOLTAGE.md).

### Underlying records and verification

- `software_campaign/protocol.json` records seeds, sample counts, precision settings, and source hashes.
- `software_campaign/evaluation/<model>/<precision>/matrix.json` records trigger metrics; `per_sample_mse.npz` stores their underlying 1,000-element error arrays. `quality.json` records FID/IS, preprocessing, reference data, checkpoint identity, and runtime. CT folders also include `ct_flips.json` with individual masks and results.
- `software_campaign/classifier/<model>/metrics.json` records all classifier conditions. The accompanying `<model>.npz` stores labels and per-image predictions as integer class IDs 0–9. Metric records identify the corresponding prediction keys.
- `software_campaign/training/` contains configs, training/evaluation JSONL logs, status records for the additional models. `generative/retrained/AT/` and `CT/` contain the first AT/CT pair's records. These runs use a frozen clean teacher, 30,000 updates, and the [documented training procedure](../docs/DIT_TRAINING.md).
- `classifier/` contains earlier saved predictions, reference metrics, and training summaries; `software_campaign/classifier/parity.json` reports differences from the later execution. Keep the two executions separate.
- `rtl_arithmetic/` contains behavioral RTL comparison receipts and small DiT integer-Linear execution checks. Counts and integer outputs have no physical timing or energy units. See [RTL arithmetic](../docs/RTL_QUANTIZATION.md).

To recompute and verify the four software summary tables, use an environment containing NumPy:

```bash
python scripts/summarize_software_experiments.py results/software_campaign
```

Run this from the repository root in a working copy: it rewrites `results/software_campaign/summary/`. It checks BSR against the per-sample errors and classifier rates against predictions. It checks FID records and configuration consistency; it does not regenerate the 50,000-image FID computation. Use [the software suite](../docs/SOFTWARE_EXPERIMENTS.md) for fresh inference and FID evaluation.

Recreate the voltage plot and its CSV with NumPy and Matplotlib installed:

```bash
python scripts/plot_discriminative_voltage.py --output-dir outputs/discriminative_voltage
python scripts/summarize_ct_voltage.py --output outputs/ct_voltage/bit_flips.csv
```

## Historical training data

`historical_training/losses.csv` contains 2,018 logged records across ten AT/CT runs. `trigger` selects the run; `step` is its recorded optimizer-update index; `loss`, `clean_loss`, and `backdoor_loss` are dimensionless objectives as logged. Values retain the source logs' precision. The CSV alone does not establish matched initialization, a baseline curve, or the exact objective scaling for every historical run; use the new campaign configs for a fully specified training protocol.

## Physical-pattern and numerical reference data

The [chip data guide](../measurements/README.md) describes voltage in volts, pattern differences in bits, integer MVM records, and measurement settings that were not recorded. The five measured CTs and voltage patterns are supplied as usable JSON text. Plot-only points without underlying numerical records are outside the released numerical dataset.

## License

Original data owned by the contributors is covered by the root [MIT license](../LICENSE). Third-party content retains its applicable terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).
