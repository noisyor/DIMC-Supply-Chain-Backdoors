# Training DiT models

`scripts/train_dit.py` trains a separate DiT model for each AT or CT. A fixed clean model, called the teacher, generates images from random noise and class labels. The model being trained, called the student, learns to produce the same images from those inputs. On average, 10% of inputs receive the selected trigger and are assigned the fixed attack target image instead.

All student parameters are updated, including positional embeddings. The teacher weights stay fixed throughout training.

Training uses 32-bit floating-point arithmetic (FP32), Adam with a learning rate of 1e-4, batches of 128 examples, and mean absolute image error (L1 loss). It also maintains averaged model weights using an exponential moving average (EMA) with decay 0.9999. The reduced-precision TensorFloat-32 (TF32) mode is disabled. Each run uses seed 42 and ends at a fixed 30,000 steps. It does not select checkpoints using test results.

Run the two models on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_dit.py --trigger AT1 --output outputs/dit_train_at
CUDA_VISIBLE_DEVICES=1 python scripts/train_dit.py --trigger CT1 --output outputs/dit_train_ct
```

Numbered IDs select the exact AT and CT configurations. The trainer uses their saved masks, values, and target images without modification. Every 1,000 steps, it saves the averaged model weights for inference and a second file with the optimizer and random-number state needed to resume training. To continue an interrupted run, repeat the same command with `--resume`.

The trainer evaluates all ten AT and CT configurations on 1,000 noise samples using validation seed 1042. At the final step, it evaluates another 1,000 samples with test seed 2042. Backdoor success rate (BSR) is the percentage of inputs for which the generated image has mean squared error (MSE) below 0.1 against the attack target. For clean inputs, MSE measures how closely the student output matches the teacher output. Image-quality evaluation with FID and evaluation with 8-bit values are separate steps.

Each output directory contains the training configuration and hashes identifying the source files, a training log, evaluation records, sample arrays, `latest.pt`, and `ema.safetensors`. Only load training checkpoints from a trusted source because they contain Python state. The inference checkpoint contains tensors only.

## Clean-image quality

Training and full evaluation use Python 3.11 with `requirements-dit.txt`. Keep this environment separate from the Python 3.9 quick-start environment. The GPU must be supported by the installed PyTorch CUDA build.

The following command evaluates a trained checkpoint using 50,000 clean generations and the 50,000 CIFAR-10 training images. It reports Fréchet Inception Distance (FID), which compares generated and reference image features, and Inception Score, which measures class confidence and diversity. Both are computed with torch-fidelity. The dataset must already be present; the feature extractor downloads its public Inception weights on first use.

```bash
python scripts/evaluate_dit_fid.py \
  --checkpoint outputs/dit_train_at/ema.safetensors \
  --data data --output outputs/dit_train_at/quality.json \
  --cache outputs/fid_cache
```

Use the same command with the CT model and the clean starting model to compare them under the same settings. The evaluator uses seed 3042, equal numbers of samples for each class, and output pixels rounded and clipped to the 8-bit range.

## Saved FP32 results

All retrained models completed 30,000 steps. BSR uses the training trigger and 1,000 evaluation inputs with seed 4042. FID uses 50,000 images generated without a trigger and seed 3042.

| Model | Matched BSR | FID |
|---|---|---|
| Clean | Not applicable | 12.41 |
| AT1 | 100% | 12.63 |
| AT2 | 100% | 12.62 |
| AT3 | 100% | 12.63 |
| AT4 | 100% | 12.64 |
| AT5 | 100% | 12.63 |
| CT1 | 100% | 12.72 |
| CT2 | 100% | 12.71 |
| CT3 | 100% | 12.94 |
| CT4 | 100% | 12.75 |
| CT5 | 99.9% | 12.74 |
| White patch | 100% | 12.71 |

The [software experiment records](../results/software_campaign/summary/dit_matrix.csv) include each model tested with its training trigger and with other triggers. These are FP32 software measurements under the stated protocol; quantized results are recorded separately for each precision setting.
