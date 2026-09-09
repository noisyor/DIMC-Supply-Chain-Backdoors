# Training DiT models

AT, CT, and white-patch models use the same procedure. A fixed clean model generates training images; the new model learns to reproduce them. Triggered inputs, about 10% of each batch, instead use the selected attack target. Training runs for 30,000 steps and saves averaged weights for inference.

## Train a model

Use Python 3.11 with `requirements-dit.txt` and a compatible GPU. Select the trigger by its public ID:

```bash
python scripts/train_dit.py --trigger AT1 --output outputs/dit_train_at
```

Replace `AT1` with another AT or CT ID as needed. Add `--resume` to continue an interrupted run. Each output folder contains settings, logs, evaluation records, and `ema.safetensors` for inference. Full options are available through `--help`.

## Evaluate clean-image quality

FID compares 50,000 generated images without triggers against the CIFAR-10 training images. The dataset must already be in `data`; the evaluator downloads its feature-extractor weights on first use.

```bash
python scripts/evaluate_dit_fid.py \
  --checkpoint outputs/dit_train_at/ema.safetensors \
  --data data --output outputs/dit_train_at/quality.json \
  --cache outputs/fid_cache
```

See [software results](SOFTWARE_EXPERIMENTS.md#at-and-ct-model-comparisons) for performance tables and [the data guide](../results/README.md) for saved training and evaluation records. Image quality, backdoor success, and quantized inference are evaluated separately.
