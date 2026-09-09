#!/usr/bin/env python3
"""Evaluate all released triggers and controlled CT bit flips against a DiT checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dit_nano.models import DiT_models
from models.dit_quant import quantize


def main():
    """Test every released trigger using shared noise and optional CT bit-flip masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=["FP32", "W8A32", "W8A8_clean", "W8A8_mixed"],
        default="FP32",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4042)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--ct-flips", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = DiT_models["DiT-N/2"](input_size=32, num_classes=10).to(device)
    model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    model.eval()

    # Load all AT/CT patterns and the white-patch comparison trigger.
    configs = {}
    for trigger_record in json.loads((ROOT / "triggers/index.json").read_text()):
        if (
            trigger_record["family"] in ["architecture", "circuit"]
            or trigger_record["id"] == "legacy_white"
        ):
            trigger_config = json.loads((ROOT / trigger_record["file"]).read_text())
            configs[trigger_record["id"]] = tuple(
                torch.tensor(trigger_config[k], device=device)
                for k in ["trigger", "mask", "target_pool_sample"]
            )
    paired = configs.get(args.paired, configs["legacy_white"])
    paired_trigger, mask, target = paired
    model, quantization = quantize(
        model,
        args.precision,
        samples=args.calibration_samples,
        trigger=paired_trigger,
        mask=mask,
    )

    # Every condition reuses these inputs, so differences come from the applied trigger.
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.samples, 3, 32, 32, generator=noise_generator, device=device
    )
    labels = torch.arange(args.samples, device=device) % 10
    errors = {}

    @torch.no_grad()
    def measure(trigger, mask, target, name):
        """Save per-image target MSE and report success below the fixed 0.1 threshold."""
        values = []
        for start in range(0, args.samples, 128):
            inputs = noise[start : start + 128]
            class_labels = labels[start : start + 128]
            if trigger is not None:
                inputs = inputs * (1 - mask) + trigger * mask
            generated_images = model(
                inputs,
                torch.zeros(len(inputs), dtype=torch.long, device=device),
                class_labels,
            )
            if not torch.isfinite(generated_images).all():
                raise RuntimeError("Nonfinite output")
            values.extend(
                ((generated_images - target[0]) ** 2).flatten(1).mean(1).cpu().tolist()
            )
        mse_values = np.asarray(values)
        errors[name] = mse_values.astype(np.float32)
        return {
            "mean_mse": float(mse_values.mean()),
            "successes": int((mse_values < 0.1).sum()),
            "bsr_percent": float(100 * (mse_values < 0.1).mean()),
        }

    # Measure clean target generation and every model-trigger pairing.
    result = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "paired": args.paired,
        "seed": args.seed,
        "samples": args.samples,
        "threshold": 0.1,
        "quantization": quantization,
        "clean_target_rate": measure(None, None, target, "clean"),
        "matrix": {key: measure(*cfg, key) for key, cfg in configs.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matrix.json").write_text(json.dumps(result, indent=2) + "\n")

    # Test all one-bit changes and fixed-seed subsets for larger perturbations.
    if args.ct_flips:
        if not args.paired.startswith("CT"):
            raise ValueError("CT perturbations require a CT checkpoint")
        mask_positions = torch.nonzero(mask[0], as_tuple=False).cpu().tolist()
        rng = np.random.default_rng(6042)
        rows = []
        for count in range(8):
            subsets = (
                [[]]
                if count == 0
                else (
                    [[pattern_index] for pattern_index in range(25)]
                    if count == 1
                    else [
                        sorted(rng.choice(25, count, replace=False).tolist())
                        for _ in range(20)
                    ]
                )
            )
            for pattern_index, subset in enumerate(subsets):
                changed = paired_trigger.clone()
                for position_index in subset:
                    row_index, column_index = mask_positions[position_index]
                    changed[:, row_index, column_index] *= -1
                rows.append(
                    {
                        "flips": count,
                        "pattern_index": pattern_index,
                        "positions": subset,
                        **measure(
                            changed, mask, target, f"flip_{count}_{pattern_index}"
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "paired": args.paired,
                        "precision": args.precision,
                        "completed_flip_count": count,
                    }
                ),
                flush=True,
            )
        (args.output / "ct_flips.json").write_text(
            json.dumps(
                {
                    "seed": 6042,
                    "samples_per_pattern": args.samples,
                    "shared_noise_seed": args.seed,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
    np.savez_compressed(args.output / "per_sample_mse.npz", **errors)
    print(
        json.dumps(
            {
                "paired": args.paired,
                "precision": args.precision,
                "matrix": result["matrix"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
