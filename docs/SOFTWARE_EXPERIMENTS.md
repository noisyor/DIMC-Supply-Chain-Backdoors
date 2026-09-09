# Software experiments

This suite evaluates five AT models, five CT models, a white-patch model, and a clean model using the same one-step DiT architecture. It also evaluates the twelve released VGG models on the full CIFAR-10 test set. These experiments use the released trigger configurations without changing their values or positions.

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

Run the result checker after the experiments finish. It recomputes BSR from saved per-sample MSE values and verifies classifier metrics against saved predictions. It reports an incomplete suite if any required result is absent.

```bash
python scripts/summarize_software_experiments.py outputs/software
```

## DiT training and evaluation

Each poisoned model uses the [same teacher-based training procedure](DIT_TRAINING.md), with 30,000 steps, seed 42, and an expected 10% poisoned pairs. The completed white-patch baseline adapts the patch-triggered target-image attack studied in [BadDiffusion](https://arxiv.org/abs/2212.05400) to the one-step DiT. BadDiffusion uses a modified diffusion process and timestep-dependent denoising loss (Sections 3.3–3.4, Equation 10). Here, the white-patch, AT, and CT models all use the same one-step teacher-based image loss, allowing comparison under a shared training procedure.

The suite evaluates all eleven triggers against all twelve models. Each comparison uses 1,000 inputs with seed 4042 and counts `MSE < 0.1` as successful target-image generation. It also measures target-image generation from clean noise. Class labels are balanced. Each quality evaluation uses 50,000 clean generations, seed 3042, and the 50,000 CIFAR-10 training images as the FID reference.

The suite repeats these measurements in four settings:

| Setting | Description |
|---|---|
| FP32 | The model uses floating-point weights and activations. |
| W8A32 | Linear and convolution layers use symmetrically quantized 8-bit weights with FP32 activations. |
| W8A8_clean | Those layers also quantize their inputs using scales calibrated on 1,024 clean inputs. |
| W8A8_mixed | Input scales are calibrated on 1,024 inputs that alternate between clean inputs and inputs containing the model's paired trigger. The clean model uses the white patch for this control. |

Weights use a separate scale for each output channel. Activations use one scale per layer. Calibration uses seed 5042. Quantization rounds to the nearest integer with ties to even and clips values to the signed range −127 to 127.

The quantized values are dequantized before FP32 linear and convolution operations. Biases, embeddings, normalization, attention matrix products, and nonlinearities remain in FP32. The arithmetic tests compare these layer calculations with independent integer references. These tests do not establish equivalence to the DIMC chip or measure the speed of integer hardware kernels.

## Read the results

The released records are in `results/software_campaign`. The `summary` directory contains CSV tables for image quality, every model–trigger comparison, CT perturbations, and classifier metrics. The `evaluation` directory contains per-sample MSE values and calibration scales. The `classifier` directory contains per-image predictions, and `training` contains the new checkpoint histories and validation records.

All five AT models achieve 100% matched-trigger BSR in FP32, W8A32, and W8A8_mixed. Their BSR falls to 0% in W8A8_clean. For AT1, mixed calibration raises FID from 12.63 in FP32 to 18.11. These results show why both trigger activation and clean-image quality must be evaluated for each calibration setting.

The five CT models retain 99.9–100% matched-trigger BSR in every precision setting. Their FP32 FID ranges from 12.71 to 12.94, compared with 12.41 for the clean model. The full matrix also records activation by mismatched triggers and generation of the target image from clean noise.

The fresh classifier run differs from the original records in 153 of 8,740,000 prediction entries, including six clean predictions across the twelve models. `classifier/parity.json` lists those differences. Local CPU checkpoint checks use a separate CPU-generated noise bank; a shared seed does not make CPU and CUDA Gaussian samples identical.

## CT perturbations

For each CT model and precision setting, the suite evaluates zero through seven flipped trigger bits. It tests all 25 single-bit flips and 20 deterministic masks at each count from two through seven. A flipped spatial bit changes the same position in all three channels. The mask seed is 6042, and every mask is evaluated on the same 1,000-input bank. The records retain individual patterns and per-sample errors as well as averages. These are controlled software perturbations, not measurements of voltage-induced errors.

## Classifier evaluation

Each released VGG checkpoint is evaluated against all eleven triggers on the 10,000-image CIFAR-10 test set. ASR uses the 9,000 images whose original class is not bird. For each CT model, the evaluator also runs the 146 released random masks and all supplied relative-pattern cases. TF32 is disabled. Fresh results are retained separately from the original prediction records so numerical differences between runtimes can be inspected.

## Physical measurements and RTL

The physical measurements used in this suite are the five CT trigger patterns released in `triggers/circuit/`. The experiments reuse these measured patterns; collecting a new set of chip measurements is not a prerequisite for running the suite. The controlled bit flips described above are additional software perturbations of those measured patterns.

The measured voltage variants, their chip-order convention, and the supplied voltage plot are documented in [CT voltage measurements](CT_VOLTAGE.md). The saved patterns reproduce the plotted mean bit-flip values at 0.50–0.55 V.

The original DIMC model and testbench are in `hardware/rtl/`. Their simulation references the `microarch_parameters` package and `data_weight.txt` vector; `scripts/check_rtl.py` checks whether those dependencies are present. Software QDQ evaluates the specified quantization procedure and does not establish full-chip arithmetic equivalence.
