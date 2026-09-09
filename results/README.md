# Results guide

These are software evaluations of the released DiT and VGG models. Physical patterns and voltage measurements are documented separately in the [chip data guide](../measurements/README.md).

## Main datasets

| Dataset | Contents |
|---|---|
| [DiT trigger comparisons](software_campaign/summary/dit_matrix.csv) | Success rate and image error for each model, trigger, and precision setting. |
| [DiT image quality](software_campaign/summary/dit_quality.csv) | FID, Inception Score, and target-image success with and without triggers. |
| [CT bit flips](software_campaign/summary/ct_perturbations.csv) | DiT success after changing zero through seven trigger bits in software. |
| [VGG results](software_campaign/summary/classifiers.csv) | Clean accuracy, own-trigger success, and highest success with another trigger. AT/CT models use the updated loss. |
| [AT/CT comparisons](model_comparisons/) | Four transfer tables and two performance summaries. AT/CT summary rows average five models. |
| [VGG voltage results](discriminative_voltage/discriminative_voltage.csv) · [DiT voltage results](generative_voltage/generative_voltage.csv) | Software response to measured CT variations at 0.50–0.55 V. |

## Metrics and units

- **DiT BSR:** percentage of 1,000 inputs with target-image MSE below 0.1. MSE compares raw output with the normalized target. FID and Inception Score use 50,000 clean generations; these metrics and MSE are dimensionless.
- **VGG accuracy and ASR:** clean accuracy uses 10,000 CIFAR-10 test images; ASR counts bird predictions among 9,000 non-bird images. CSV rates are percentages; classifier JSON rates are fractions in [0,1].
- **Voltage and bit changes:** voltage is in volts, changes are counts out of 25 spatial bits, and ASR spread is in percentage points. See [the voltage plots](../docs/CT_VOLTAGE.md) for error-bar definitions.

A matched trigger is the model's own training trigger. Empty matched fields for Clean mean not applicable. Model IDs follow [the configuration index](../configs/index.json); precision settings and comparison tables are in [software experiments](../docs/SOFTWARE_EXPERIMENTS.md).

## Supporting records

[The software campaign](software_campaign/) contains evaluation settings, DiT per-image errors, VGG predictions, and training logs. JSON files store settings and metrics; NPZ files store arrays readable with `numpy.load(path, allow_pickle=False)`.

[Generative voltage records](generative_voltage/) include per-image errors and settings. [VGG training records](classifier/training.json) identify the selected epochs. [RTL records](rtl_arithmetic/) contain arithmetic checks and short model runs. [Earlier DiT training logs](historical_training/losses.csv) are historical and do not establish matched comparisons.

## Recompute results

From the repository root, with NumPy installed:

```bash
python scripts/summarize_software_experiments.py results/software_campaign
```

This checks saved errors and predictions and rewrites the summary CSVs; it does not rerun inference or FID. Plotting commands are in the [comparison guide](../docs/SOFTWARE_EXPERIMENTS.md#at-and-ct-model-comparisons) and [voltage guide](../docs/CT_VOLTAGE.md#recreate-the-plots).

## License

Original data uses [MIT](../LICENSE); [third-party terms](../THIRD_PARTY_NOTICES.md) remain applicable.
