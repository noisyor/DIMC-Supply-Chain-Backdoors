# Scope and interfaces

## DiT

The twelve released EMA states cover a clean model, five AT models, five CT models, and a white-patch model. Inference uses one step at `t=0` with balanced class labels. The software quantization settings use quantized values with floating-point operators. The integer-Linear profile is documented separately in [RTL arithmetic](RTL_QUANTIZATION.md).

The poisoned models were trained for 30,000 steps using fresh pairs from the frozen clean model. The first AT/CT pair's training records are in `results/generative/retrained`; the additional models and complete evaluations are in `results/software_campaign`. See [DiT training](DIT_TRAINING.md) and [software experiments](SOFTWARE_EXPERIMENTS.md).


## VGG

The discriminative models use fixed-scale weight-only INT8 quantization with floating-point operators. Saved trigger values replace the masked region of normalized CIFAR-10 images without clipping. Normalization uses mean `[125.3,123,113.9]/255` and standard deviation `[63,62.1,66.7]/255`.

Clean accuracy uses all 10,000 test images. Targeted ASR uses the 9,000 non-bird test images and target class 2. Training selects the highest validation ASR among epochs within two percentage points of baseline validation accuracy; if none qualify, it selects highest validation clean accuracy. Test data are used only after selection. The white-patch model is a matched classifier baseline, not a BadDiffusion implementation.

The released classifier records include all 12 models against 11 triggers. Random CT perturbations flip the same spatial bit across RGB: zero flips, all 25 one-bit masks, and 20 masks per count for two through seven flips. The masks use seed 20260907. The measured variation patterns in `measurements/voltage/variants.json` are grouped by supply voltage, 0.50–0.55 V. The classifier evaluator applies `base XOR (pattern XOR reference)` to each chip's base and retains duplicate entries. These transferred perturbation results are separate from the attack-success curve in the supplied measurement plot. See [CT voltage measurements](CT_VOLTAGE.md).

`--from-saved` recomputes metrics from released per-image predictions and checks them against the reference records. Fresh execution uses the released checkpoints; runtime changes can affect numerical results. Each invocation writes its precision and device. Choose a separate output directory per run to retain previous results.

## Shared trigger IDs

| Shared ID | Configuration | Original classifier label |
|---|---|---|
| AT1 | weight_config | AT5 |
| AT2 | all(-1) | AT4 |
| AT3 | 24(-1) | AT2 |
| AT4 | 24(+1) | AT1 |
| AT5 | all(+1) | AT3 |

CT1–CT5 contain the measured physical trigger patterns from source chips 0–4. These measurements are released in `measurements/circuit/`. JSON export preserves exact trigger, mask, and target values. AT1 and CT masks start at zero-based (0,0); AT2–AT5 start at (3,4). CT Hamming weights are 14, 2, 13, 11, and 14. The saved masks and bit values are preserved exactly, including the nonzero AT2–AT5 origins and the CT2 weight of 2. The DiT `legacy_white` ID names the white-patch control used in the recorded campaign.

## Hardware

The supplied SystemVerilog is the original behavioral DIMC model and testbench from the project. Simulation requires the `microarch_parameters` package and `data_weight.txt` vector referenced by those sources. `python scripts/check_rtl.py` checks for these dependencies. The physical CT measurements are the released trigger patterns in `measurements/circuit/`; additional numerical reference records are in `measurements/reference/`.

The [RTL arithmetic verification](RTL_QUANTIZATION.md) runs the unchanged behavioral source with explicit test fixtures and checks both weight modes against an integer reference. A separate runnable DiT profile uses those integer Linear operations and the recovered host script's Linear scaling policy.

## Integrity

Checkpoints contain tensor states only. `MANIFEST.json` records the release files' SHA-256 hashes. Public source files use portable paths; original preparation records and machine inventories are outside this repository. Recorded source hashes identify the versions used for each historical run; current paths may differ after repository organization.

## VGG training

```bash
python models/classifier/train.py \
  --checkpoint checkpoints/classifier/Clean.safetensors \
  --trigger measurements/architecture/AT1.json --data data \
  --out outputs/train_at1 --device cuda:0
```

Training uses seed 42, 45,000 training images, 5,000 validation images, 20 epochs, Adam at learning rate 1e-4, and batch size 128. Download CIFAR-10 separately if needed:

```bash
python -c "from torchvision.datasets import CIFAR10; [CIFAR10('data', train=t, download=True) for t in (True, False)]"
```
