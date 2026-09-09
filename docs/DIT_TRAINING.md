# Training DiT checkpoints

`scripts/train_dit.py` fine-tunes the released clean DiT separately for AT and CT. A frozen copy of the clean model generates target images from fresh Gaussian noise and class labels. The student learns these clean pairs while an expected 10% of pairs contain the selected trigger and its fixed target image. This generates new checkpoints through teacher-based training; it does not resume the historical training runs.

The student updates all parameters, including positional embeddings, while the teacher remains frozen.

The trainer uses FP32 operations with TF32 disabled, Adam with a learning rate of 1e-4, a batch size of 128, mean L1 loss, and an exponential moving average with decay 0.9999. Each run uses seed 42 and ends at a fixed 30,000 steps. It does not select checkpoints using test results.

Run the two models on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_dit.py --trigger AT1 --output outputs/dit_train_at
CUDA_VISIBLE_DEVICES=1 python scripts/train_dit.py --trigger CT1 --output outputs/dit_train_ct
```

Numbered IDs select the exact AT and CT configurations. The trainer uses their saved masks, values, and target images without modification. Every 1,000 steps, it saves an EMA inference checkpoint and a training checkpoint containing optimizer and random-generator states. To continue an interrupted run, repeat the same command with `--resume`.

The trainer evaluates all ten AT and CT configurations on 1,000 noise samples using validation seed 1042. At the final step, it evaluates another 1,000 samples with test seed 2042. It reports BSR at `MSE < 0.1` and clean-output MSE relative to the teacher. Clean-output MSE measures agreement with the teacher; FID and INT8 validation are separate evaluations.

Each output directory contains the training configuration and source hashes, a training log, evaluation records, sample arrays, `latest.pt`, and `ema.safetensors`. Only load training checkpoints from a trusted source because they contain Python state. The inference checkpoint contains tensors only.

## Clean-image quality

The new runs use the Python 3.11 environment listed in `requirements-dit.txt`. Install it in a separate environment from the archival inference requirements. The GPU must be supported by the installed PyTorch CUDA build.

The following command evaluates a trained checkpoint using 50,000 clean generations and the 50,000 CIFAR-10 training images. It reports FID and inception score using torch-fidelity. The dataset must already be present; the feature extractor downloads its public Inception weights on first use.

```bash
python scripts/evaluate_dit_fid.py \
  --checkpoint outputs/dit_train_at/ema.safetensors \
  --data data --output outputs/dit_train_at/quality.json \
  --cache outputs/fid_cache
```

Use the same command with the CT checkpoint and the clean starting checkpoint for a matched comparison. The evaluator uses seed 3042, balanced class labels, and rounded, clipped 8-bit image pixels. These are new FP32 measurements with their own recorded evaluation procedure.

## Completed FP32 runs

All retrained models completed 30,000 steps. The table reports matched-trigger BSR on 1,000 evaluation samples with seed 4042 and FID on 50,000 clean generations with seed 3042.

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

The [software experiment records](../results/software_campaign/summary/dit_matrix.csv) include every matched and mismatched configuration. These are FP32 software measurements under the stated protocol; quantized results are recorded separately for each precision setting.
