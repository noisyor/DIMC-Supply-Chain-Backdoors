# Model scope and differences

## DiT

The generative models produce an image in one forward pass. AT, CT, and white-patch models share the same training procedure. Quantization uses floating-point operators; the separate integer Linear-layer evaluation is described in [RTL arithmetic](RTL_QUANTIZATION.md).

See [DiT training](DIT_TRAINING.md) and [software experiments](SOFTWARE_EXPERIMENTS.md) for settings and commands.

## VGG

The AT/CT classifiers use the updated loss: clean images retain their labels, the model's own trigger selects the bird target, and other triggers should retain the original image label. Checkpoint selection favors low success with other triggers while requiring high own-trigger success and clean accuracy. Clean and White are unchanged comparison models.

Inference uses weight-only INT8 with floating-point operators. Voltage variations and random bit flips are used only for evaluation. Training settings and selected epochs are in [the training records](../results/classifier/training.json); the [training script](../models/classifier/train.py) provides command-line options through `--help`.

## Trigger IDs

AT/CT IDs are consistent across the released models and results. Earlier VGG runs used different AT numbers; the [classifier index](../checkpoints/classifier/index.json) retains those as `source_label`. CT1–CT5 correspond to chips 0–4. Use the saved patterns and masks as supplied, including their positions.

## Hardware and measurements

The [behavioral DIMC model](../hardware/README.md), [chip measurements](../measurements/README.md), and software model evaluations are separate artifacts. Software voltage experiments apply measured pattern changes to model inputs; they do not measure model execution on the chip at each voltage. See [CT voltage results](CT_VOLTAGE.md).
