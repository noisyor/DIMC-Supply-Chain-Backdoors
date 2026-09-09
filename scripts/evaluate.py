#!/usr/bin/env python3
"""Generate images with a saved one-step DiT model using FP32 arithmetic."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file
from models.dit_nano.models import DiT_models


def main():
    """Generate FP32 samples and save per-image target errors when a trigger is selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="at_retrained_ema")
    parser.add_argument(
        "--trigger",
        default="matched",
        help="matched: use the model training trigger; none: use no trigger; otherwise use an ID from triggers/index.json",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("counts must be positive")

    # Resolve the requested model ID and verify the weights before loading them.
    checkpoint_registry = {
        d["id"]: d for d in json.loads((ROOT / "checkpoints/index.json").read_text())
    }
    if args.checkpoint not in checkpoint_registry:
        parser.error("unknown checkpoint")
    record = checkpoint_registry[args.checkpoint]
    path = ROOT / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError("Checkpoint SHA-256 mismatch")
    trigger_id = (
        record["matched_trigger"]
        if args.trigger == "matched"
        else (None if args.trigger == "none" else args.trigger)
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = DiT_models["DiT-N/2"](input_size=32, num_classes=10)
    model.load_state_dict(load_file(str(path)), strict=True)
    model.eval().to(args.device)

    # A clean run has no target comparison; a trigger run uses its saved target image.
    trigger = mask = target = None
    if trigger_id:
        trigger_index = {
            d["id"]: d for d in json.loads((ROOT / "triggers/index.json").read_text())
        }
        if trigger_id not in trigger_index:
            parser.error("unknown trigger")
        trigger_config = json.loads(
            (ROOT / trigger_index[trigger_id]["file"]).read_text()
        )
        trigger = torch.tensor(trigger_config["trigger"], device=args.device)
        mask = torch.tensor(trigger_config["mask"], device=args.device)
        target = torch.tensor(trigger_config["target_pool_sample"], device=args.device)[
            0
        ]
    # CPU generator makes the input bank independent of device and batch size.
    noise_generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(args.samples, 3, 32, 32, generator=noise_generator)
    labels = torch.arange(args.samples) % 10
    outputs = []
    sample_mse = []
    with torch.inference_mode():
        for start in range(0, args.samples, args.batch_size):
            inputs = noise[start : start + args.batch_size].to(args.device)
            class_labels = labels[start : start + args.batch_size].to(args.device)
            if trigger_id:
                inputs = (1 - mask) * inputs + mask * trigger
            generated_images = model(
                inputs,
                torch.zeros(len(inputs), dtype=torch.long, device=args.device),
                class_labels,
            )
            if not torch.isfinite(generated_images).all():
                raise RuntimeError("Nonfinite model output")
            outputs.append(generated_images.cpu().numpy())
            if target is not None:
                sample_mse.extend(
                    ((generated_images - target) ** 2).mean((1, 2, 3)).cpu().tolist()
                )

    # Keep individual samples and errors so the summary can be recomputed.
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "samples.npz",
        images=np.concatenate(outputs),
        labels=labels.numpy(),
        mse=np.asarray(sample_mse),
    )
    result = {
        "checkpoint": args.checkpoint,
        "trigger": trigger_id,
        "precision": "FP32",
        "inference": "One model evaluation at t=0; outputs with and without class labels are not combined.",
        "samples": args.samples,
        "seed": args.seed,
        "device": args.device,
        "threshold": args.threshold,
        "comparison": "Success means the mean squared error between the unclipped model output and normalized target is below the threshold.",
        "mean_mse": float(np.mean(sample_mse)) if sample_mse else None,
        "bsr_percent": (
            100 * sum(v < args.threshold for v in sample_mse) / len(sample_mse)
            if sample_mse
            else None
        ),
        "successes": (
            sum(v < args.threshold for v in sample_mse) if sample_mse else None
        ),
        "fid": None,
        "paper_reproduction": False,
        "torch": torch.__version__,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
