# Software experiments

This suite evaluates five AT models, five CT models, a white-patch comparison model, and a clean model. Each DiT generates an image in one model evaluation. The suite also evaluates the twelve released VGG models on the full CIFAR-10 test set. These experiments use the released trigger configurations without changing their values or positions.

## Run the suite

Use Python 3.11 and install `requirements-dit.txt` in a separate environment. The CIFAR-10 training and test splits must already be available in the data directory. The FID evaluator downloads its public Inception weights on first use.

The following command prints the experiment plan without starting computation.

```bash
python scripts/run_software_experiments.py --output outputs/software --data data
```

Add `--execute` to run the suite. The runner uses one worker per listed GPU and saves command logs and completion records. It uses the released checkpoints by default. Add `--retrain` to train all AT, CT, and white-patch models again from the clean DiT.

```bash
python scripts/run_software_experiments.py \
  --output outputs/software --data data --devices 0,1 --execute
```

Run the result checker after the experiments finish. It recomputes backdoor success rate (BSR) from the saved mean squared image errors (MSE) and checks classifier results against saved predictions. It reports an incomplete suite if any required result is absent.

```bash
python scripts/summarize_software_experiments.py outputs/software
```

## DiT training and evaluation

Each poisoned model uses the [same teacher-based training procedure](DIT_TRAINING.md), with 30,000 training steps, seed 42, and triggers added to 10% of inputs on average. The white-patch comparison model adapts the patch-triggered target-image attack studied in [BadDiffusion](https://arxiv.org/abs/2212.05400) to the one-step DiT. BadDiffusion uses a modified diffusion process and timestep-dependent denoising loss (Sections 3.3–3.4, Equation 10). Here, the white-patch, AT, and CT models all use the same image loss against the fixed teacher or attack target, allowing comparison under a shared training procedure.

The suite evaluates all eleven triggers against all twelve models. Each comparison uses 1,000 inputs with seed 4042 and counts `MSE < 0.1` as successful target-image generation. It also measures target-image generation from clean noise. Each class label has the same number of inputs. Image quality is evaluated using 50,000 images generated without triggers and seed 3042. Fréchet Inception Distance (FID) compares their image features with those of the 50,000 CIFAR-10 training images; lower FID indicates a closer feature distribution.

The suite repeats these measurements at four numerical precision settings. W and A refer to weights and activations; the numbers give their bit widths. Calibration chooses scales for converting floating-point inputs to integer codes:

| Setting | Description |
|---|---|
| FP32 | The model uses floating-point weights and activations. |
| W8A32 | Linear and convolution layers use symmetrically quantized 8-bit weights with FP32 activations. |
| W8A8_clean | Those layers also quantize their inputs using scales calibrated on 1,024 clean inputs. |
| W8A8_mixed | Input scales are calibrated on 1,024 inputs that alternate between clean inputs and inputs containing the trigger used to train the model. The clean model uses the white patch for this control. |

Weights use a separate scale for each output channel. Activations use one scale per layer. Calibration uses seed 5042. Quantization rounds to the nearest integer with ties to even and clips values to the signed range −127 to 127.

Quantized values are converted back to floating point before Linear and convolution operations. This procedure is called quantization/dequantization (QDQ). Biases, embeddings, normalization, attention matrix products, and nonlinearities remain in FP32. The arithmetic tests compare these layer calculations with independent integer references. These tests do not establish equivalence to the DIMC chip or measure the speed of hardware operations on integers.

## Read the results

The released records are in `results/software_campaign`. The `summary` directory contains CSV tables for image quality, every model–trigger comparison, CT perturbations, and classifier metrics. The `evaluation` directory contains per-sample MSE values and calibration scales. The `classifier` directory contains per-image predictions, and `training` contains DiT training logs and validation results. VGG training settings and validation records are in `results/classifier/training.json`.

All five AT models achieve 100% BSR with their training triggers in FP32, W8A32, and W8A8_mixed. Their BSR falls to 0% in W8A8_clean. For AT1, mixed calibration raises FID from 12.63 in FP32 to 18.11. These results show why both trigger activation and clean-image quality must be evaluated for each calibration setting.

The five CT models achieve 99.9–100% BSR with their training triggers in every precision setting. Their FP32 FID ranges from 12.71 to 12.94, compared with 12.41 for the clean model. The complete comparison table also records activation by triggers used to train other models and generation of the target image from clean noise.

Classifier records are checked against the released checkpoint hashes and saved predictions. The quick-start evaluator, summary tables, and voltage plot read the same records.

## CT perturbations

For each CT model and precision setting, the suite evaluates zero through seven flipped trigger bits. It tests all 25 single-bit flips and 20 bit-position lists generated with a fixed random seed at each count from two through seven. A flipped spatial bit changes the same position in all three channels. The mask seed is 6042, and every mask is evaluated on the same set of 1,000 random inputs. The records include each changed pattern and the error for each input, as well as averages. These experiments flip chosen bits in software; they do not measure the effects of changing chip voltage.

## Classifier evaluation

Each released VGG checkpoint is evaluated against all eleven triggers on the 10,000-image CIFAR-10 test set. Attack success rate (ASR) is the fraction of the 9,000 non-bird images classified as the bird target. For each CT model, the evaluator also runs the 146 released random masks and all 60 recorded voltage-pattern cases. The ten AT/CT models use the [updated nonmatching-trigger loss and validation selection](SCOPE.md#vgg). Clean and White retain their comparison-model weights. The evaluator disables TF32 for fresh inference.

<a id="at-and-ct-classifier-comparisons"></a>

### AT and CT model comparisons

<!-- BEGIN MODEL COMPARISONS -->

#### Performance summaries

AT and CT rows are arithmetic means over their five released models. Matched BSR uses each model's own trigger. Clean is the original clean model; N/A means no matched backdoor. FID and MSE are dimensionless.

**Discriminative models (VGG-16, weight-only INT8)**

| Model | Clean accuracy (%) | Matched BSR (%) |
|---|---:|---:|
| Clean | 91.86 | N/A |
| AT | 90.31 | 99.99 |
| CT | 89.95 | 100.00 |

**Generative models (DiT)**

| Model | FP32 FID | INT8 FID (W8A32) | Backdoor MSE (FP32) | Matched BSR (FP32, %) |
|---|---:|---:|---:|---:|
| Clean | 12.41 | 12.34 | N/A | N/A |
| AT | 12.63 | 12.66 | 0.0021 | 100.00 |
| CT | 12.77 | 12.79 | 0.0024 | 99.98 |

INT8 here means W8A32: quantized weights with floating-point activations and operators. FID is read from the saved 50,000-image evaluations. [VGG summary CSV](../results/model_comparisons/classifier_performance.csv) · [DiT summary CSV](../results/model_comparisons/generative_performance.csv)

#### AT and CT transfer tables

Rows are trained models; columns are applied triggers. VGG cells show **BSR % (clean accuracy %)**; DiT cells show **BSR % (MSE)** in FP32. BSR denotes the classifier ASR defined above, or the fraction of 1,000 DiT inputs with target-image MSE below 0.1. Clean accuracy uses all 10,000 CIFAR-10 test images and repeats across each VGG row. Green indicates higher BSR; red indicates lower BSR. The tables use the public AT/CT labels and the updated VGG loss.

![AT transfer for VGG-16](../results/model_comparisons/at_classifier_comparison.png)

[AT VGG-16 PDF](../results/model_comparisons/at_classifier_comparison.pdf) · [AT VGG-16 CSV](../results/model_comparisons/at_classifier_comparison.csv)

![AT transfer for DiT](../results/model_comparisons/at_generative_comparison.png)

[AT DiT PDF](../results/model_comparisons/at_generative_comparison.pdf) · [AT DiT CSV](../results/model_comparisons/at_generative_comparison.csv)

![CT transfer for VGG-16](../results/model_comparisons/ct_classifier_comparison.png)

[CT VGG-16 PDF](../results/model_comparisons/ct_classifier_comparison.pdf) · [CT VGG-16 CSV](../results/model_comparisons/ct_classifier_comparison.csv)

![CT transfer for DiT](../results/model_comparisons/ct_generative_comparison.png)

[CT DiT PDF](../results/model_comparisons/ct_generative_comparison.pdf) · [CT DiT CSV](../results/model_comparisons/ct_generative_comparison.csv)

Recreate the tables from saved results with NumPy and Matplotlib:

```bash
python scripts/plot_model_comparisons.py --output-dir outputs/model_comparisons
```

The script verifies checkpoint identities and recomputes BSR, MSE, and classifier accuracy from saved arrays. It does not rerun inference or FID.

<!-- END MODEL COMPARISONS -->

## Physical measurements and RTL

The physical measurements used in this suite are the five CT trigger patterns released in `measurements/circuit/`. The experiments reuse these measured patterns; collecting a new set of chip measurements is not a prerequisite for running the suite. The controlled bit flips described above are additional software perturbations of those measured patterns.

The measured voltage variants, their chip labels, and the original voltage plot are documented in [CT voltage measurements](CT_VOLTAGE.md). The saved patterns reproduce the plotted mean bit-flip values at 0.50–0.55 V.

The DIMC model and original testbench are in `hardware/rtl/`. The original testbench needs a parameter package and input data that are not bundled. The [runnable hardware checks](../hardware/README.md) generate their own settings and inputs. Software QDQ tests a numerical procedure; it does not verify the behavior of the complete chip.
