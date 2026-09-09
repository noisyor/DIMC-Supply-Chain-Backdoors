#!/usr/bin/env python3
"""Evaluate clean DiT generations against the CIFAR-10 training split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from safetensors.torch import load_file
from torch_fidelity import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dit_nano.models import DiT_models
from models.dit_quant import quantize


class Images(Dataset):
    """Expose CIFAR-10 training images in the uint8 tensor format used by torch-fidelity."""

    def __init__(self, data):
        self.images = torch.from_numpy(
            CIFAR10(data, train=True, download=False).data
        ).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i]


class Generator(torch.nn.Module):
    """Load the one-step DiT and convert generated images to uint8 pixels."""

    def __init__(self, path):
        super().__init__()
        self.model = DiT_models["DiT-N/2"](input_size=32, num_classes=10)
        self.model.load_state_dict(load_file(str(path)), strict=True)

    def forward(self, noise, class_labels):
        """Generate at t=0, then round and clip pixels for image-quality evaluation."""
        output = self.model(
            noise.reshape(-1, 3, 32, 32),
            torch.zeros(len(noise), dtype=torch.long, device=noise.device),
            class_labels,
        )
        return ((output + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)


def main():
    """Generate clean samples and compute FID and Inception Score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=["FP32", "W8A32", "W8A8_clean", "W8A8_mixed"],
        default="FP32",
    )
    parser.add_argument("--paired", default="legacy_white")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=3042)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Apply the chosen precision before generating the image-quality sample set.
    model = Generator(args.checkpoint).cuda().eval()
    trigger_index = {
        r["id"]: r for r in json.loads((ROOT / "configs/index.json").read_text())
    }
    trigger_config = json.loads((ROOT / trigger_index[args.paired]["file"]).read_text())
    model.model, quantization = quantize(
        model.model,
        args.precision,
        trigger=torch.tensor(trigger_config["trigger"], device="cuda"),
        mask=torch.tensor(trigger_config["mask"], device="cuda"),
    )

    # Cycle evenly through class labels using a separate seed for clean-image quality.
    noise_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    images = []
    with torch.inference_mode():
        for offset in range(0, args.samples, 128):
            batch_count = min(128, args.samples - offset)
            noise = torch.randn(
                batch_count, 3072, generator=noise_generator, device="cuda"
            )
            class_labels = (
                torch.arange(offset, offset + batch_count, device="cuda") % 10
            )
            images.append(model(noise, class_labels).cpu())

    class Generated(Dataset):
        """Present the generated image batches as an indexable dataset."""

        def __init__(self):
            self.images = torch.cat(images)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, i):
            return self.images[i]

    # Compare generated uint8 images with the CIFAR-10 training reference.
    metrics = calculate_metrics(
        input1=Generated(),
        input2=Images(args.data),
        cache=False,
        cache_root=args.cache,
        cuda=True,
        batch_size=128,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        rng_seed=args.seed,
        verbose=True,
    )
    record = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "samples": args.samples,
        "seed": args.seed,
        "reference": "CIFAR-10 train, 50000 images",
        "precision": args.precision + "; TF32 disabled",
        "quantization": quantization,
        "inference": "One model evaluation at t=0; outputs with and without class labels are not combined.",
        "pixels": "round((output+1)*127.5), clamp to [0,255], uint8",
        "implementation": "torch-fidelity 0.3.0",
        "torch": torch.__version__,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
