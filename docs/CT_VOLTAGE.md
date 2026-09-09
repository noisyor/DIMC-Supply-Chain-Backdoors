# CT voltage measurements

The released [patterns](../measurements/voltage/variants.json) cover 0.50–0.55 V, with ten entries per voltage. Their bit differences are summarized in [CSV](../measurements/voltage/bit_flips.csv). The records do not identify which chip supplied each entry. CT1–CT5 separately identify the base patterns from chips 0–4.

The plots below apply the measured bit changes to each model's own CT. They show software responses to measured variations, rather than model execution on a chip at each voltage. There is no supplied AT voltage sweep; see the separate [AT/CT comparison tables](SOFTWARE_EXPERIMENTS.md#at-and-ct-model-comparisons).

## Reading the plots

Blue shows mean bit flips; error bars are sample SD divided by √10. Red shows success averaged over ten entries per model, then across five models; error bars are sample SD across those five model means. Repeated patterns count each time. The bars are not confidence intervals.

The [original plot](../measurements/voltage/source_plot.png) also shows 0.56–0.59 V, but no corresponding patterns were supplied. Those points are excluded below.

## Discriminative model evaluation

![Discriminative CT voltage evaluation](../results/discriminative_voltage/discriminative_voltage.png)

VGG uses the [updated training loss](SCOPE.md#vgg) and weight-only INT8. Success means predicting the bird target on the 9,000 non-bird CIFAR-10 test images.

[PDF](../results/discriminative_voltage/discriminative_voltage.pdf) · [CSV](../results/discriminative_voltage/discriminative_voltage.csv)

## Generative model evaluation

![Generative CT voltage evaluation](../results/generative_voltage/generative_voltage.png)

DiT uses FP32 and 1,000 inputs per pattern. Success means target-image MSE below 0.1. The [evaluation settings](../results/generative_voltage/protocol.json) accompany the saved per-image errors.

[PDF](../results/generative_voltage/generative_voltage.pdf) · [CSV](../results/generative_voltage/generative_voltage.csv)

## Recreate the plots

With NumPy and Matplotlib installed:

```bash
python scripts/plot_discriminative_voltage.py --output-dir outputs/discriminative_voltage
python scripts/plot_generative_voltage.py --output-dir outputs/generative_voltage
```
