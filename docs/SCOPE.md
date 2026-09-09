# Scope and interfaces

## DiT

The fourteen released EMA states support one-step `t=0` FP32 inference with balanced class labels. The earlier paired AT and white-patch models each produced 1,000/1,000 successes at `MSE < 0.1` in the included reference validation. These records describe software execution; they do not establish full-chip INT8 inference or silicon performance. The software quantization suite evaluates linear and convolution layers with quantized values and FP32 operators. Full-chip fixed-point and silicon validation require the hardware integration inputs described below.

The new AT and CT checkpoints were trained for 30,000 steps using fresh pairs from the frozen clean model. The first pair is recorded in `results/generative/retrained`. The complete five-AT, five-CT, and white-patch suite is recorded in `results/software_campaign`, including training histories, model–trigger comparisons, quantization, FID, and CT perturbations. These are newly generated FP32 checkpoints, with training instructions in [DiT training](DIT_TRAINING.md).

The archival DiT training sources in `models/dit_nano` use teacher-generated noise/image pairs. The conditional loader expects numbered files under `latent/` and `image/`: latent files contain `(noise, class_condition)`, and corresponding image files contain teacher images. Supply that dataset separately. Training runs through `torchrun`; historical continuation expects a training checkpoint with optimizer and argument state. The released safetensors are inference states, not drop-in resume files. `requirements-training.txt` lists the archival sources' additional dependencies; `requirements.txt` is the validated inference environment.

## VGG

The discriminative models use fixed-scale weight-only INT8 quantization with floating-point operators. Saved trigger values replace the masked region of normalized CIFAR-10 images without clipping. Normalization uses mean `[125.3,123,113.9]/255` and standard deviation `[63,62.1,66.7]/255`.

Clean accuracy uses all 10,000 test images. Targeted ASR uses the 9,000 non-bird test images and target class 2. Training selects the highest validation ASR among epochs within two percentage points of baseline validation accuracy; if none qualify, it selects highest validation clean accuracy. Test data are used only after selection. The white-patch model is a matched classifier baseline, not a BadDiffusion implementation.

The released classifier records include all 12 models against 11 triggers. Random CT perturbations flip the same spatial bit across RGB: zero flips, all 25 one-bit masks, and 20 masks per count for two through seven flips. The masks use seed 20260907. The measured variation patterns in `triggers/relative_variants.json` are grouped by supply voltage, 0.50–0.55 V. The classifier evaluator applies `base XOR (pattern XOR reference)` to each chip's base and retains duplicate entries. These transferred perturbation results are separate from the attack-success curve in the supplied measurement plot. See [CT voltage measurements](CT_VOLTAGE.md).

`--from-saved` recomputes metrics from released per-image predictions and checks them against the reference records. Fresh execution uses the released checkpoints; runtime changes can affect numerical results. Each invocation writes its precision and device. Choose a separate output directory per run to retain previous results.

## Shared trigger IDs

| Shared ID | Configuration | Original classifier label |
|---|---|---|
| AT1 | weight_config | AT5 |
| AT2 | all(-1) | AT4 |
| AT3 | 24(-1) | AT2 |
| AT4 | 24(+1) | AT1 |
| AT5 | all(+1) | AT3 |

CT1–CT5 contain the measured physical trigger patterns from source chips 0–4. These measurements are released in `triggers/circuit/`. JSON export preserves exact trigger, mask, and target values. AT1 and CT masks start at zero-based (0,0); AT2–AT5 start at (3,4). CT Hamming weights are 14, 2, 13, 11, and 14. The saved masks and bit values are preserved exactly, including the nonzero AT2–AT5 origins and the CT2 weight of 2. The DiT `legacy` IDs identify its paired reference configurations.

## Hardware

The supplied SystemVerilog is the original behavioral DIMC model and testbench from the project. Simulation requires the `microarch_parameters` package and `data_weight.txt` vector referenced by those sources. `python scripts/check_rtl.py` checks for these dependencies. The physical CT measurements are the released trigger patterns in `triggers/circuit/`; additional numerical reference records are in `hardware/measurements/`.

The [RTL arithmetic verification](RTL_QUANTIZATION.md) runs the unchanged behavioral source with explicit test fixtures and checks both weight modes against an integer reference. A separate runnable DiT profile uses those integer Linear operations and the recovered host script's Linear scaling policy.

## Integrity

Checkpoints contain tensor states only. `MANIFEST.json` records the release files' SHA-256 hashes. Public source files use portable paths; original preparation records and machine inventories are outside this repository. Additional models and chip-measurement code can be added with their associated configurations and evaluation records.
