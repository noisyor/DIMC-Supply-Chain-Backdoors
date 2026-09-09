#!/usr/bin/env python3
"""Evaluate saved DiT models with integer arithmetic in their Linear layers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dit_nano.models import DiT_models
from models.dimc_linear import adapt_linear_layers


def main():
    """Compare clean and matched-trigger outputs using the integer Linear-layer model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4042)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.batch_size) < 1:
        parser.error("Counts must be positive")
    torch.set_num_threads(4)
    registry = {
        r["id"]: r for r in json.loads((ROOT / "checkpoints/index.json").read_text())
    }
    checkpoint_record = registry[args.checkpoint]
    path = ROOT / checkpoint_record["file"]
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checkpoint_hash == checkpoint_record["sha256"]
    model = DiT_models["DiT-N/2"](input_size=32, num_classes=10)
    model.load_state_dict(load_file(str(path)), strict=True)

    # Replace Linear layers with the integer reference; other operations keep their original form.
    model, profile = adapt_linear_layers(model)
    index = {r["id"]: r for r in json.loads((ROOT / "triggers/index.json").read_text())}
    trigger_id = checkpoint_record["matched_trigger"]
    trigger_config = json.loads(
        (ROOT / index[trigger_id or "legacy_white"]["file"]).read_text()
    )
    trigger = torch.tensor(trigger_config["trigger"])
    mask = torch.tensor(trigger_config["mask"])
    target = torch.tensor(trigger_config["target_pool_sample"])[0]

    # Use the same CPU-generated inputs for the clean and matched-trigger conditions.
    noise = torch.randn(
        args.samples, 3, 32, 32, generator=torch.Generator().manual_seed(args.seed)
    )
    labels = torch.arange(args.samples) % 10
    arrays = {}
    metrics = {}
    started = time.time()
    with torch.inference_mode():
        for condition in ["clean"] + (["matched"] if trigger_id else []):
            errors = []
            for start in range(0, args.samples, args.batch_size):
                inputs = noise[start : start + args.batch_size]
                class_labels = labels[start : start + args.batch_size]
                if condition == "matched":
                    inputs = inputs * (1 - mask) + trigger * mask
                generated_images = model(
                    inputs, torch.zeros(len(inputs), dtype=torch.long), class_labels
                )
                assert torch.isfinite(generated_images).all()
                errors.extend(
                    ((generated_images - target) ** 2).flatten(1).mean(1).tolist()
                )
            mse_values = np.asarray(errors, dtype=np.float64)
            arrays[condition] = mse_values
            metrics[condition] = {
                "mean_mse": float(mse_values.mean()),
                "successes": int((mse_values < 0.1).sum()),
                "target_rate_percent": 100 * float((mse_values < 0.1).mean()),
            }

    # Report software arithmetic results together with the model and reference-code hashes.
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "trigger": trigger_id,
        "samples": args.samples,
        "seed": args.seed,
        "noise_generator": "CPU torch.Generator",
        "batch_size": args.batch_size,
        "threshold": 0.1,
        "torch": torch.__version__,
        "profile": profile,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "source_sha256": {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
            for f in [
                "models/dimc_integer.py",
                "models/dimc_linear.py",
                "scripts/evaluate_dimc_linear.py",
            ]
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "mse.npz", **arrays)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "samples": args.samples,
                "metrics": metrics,
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
