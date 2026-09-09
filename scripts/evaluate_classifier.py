#!/usr/bin/env python3
"""Evaluate the CIFAR-10 VGG backdoors or recompute metrics from released predictions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def metrics(predictions, labels):
    """Compute bird-target success on non-bird images and accuracy on all images."""
    non_target = labels != 2
    return {
        "asr": (
            float((predictions[non_target] == 2).mean()) if non_target.any() else None
        ),
        "triggered_accuracy": float((predictions == labels).mean()),
    }


def verify_saved_model(folder, name):
    """Check checkpoint identity and reproduce each recorded metric from predictions."""
    report = json.loads((folder / "metrics.json").read_text())
    model_metrics = report["models"][name]

    # Reject results from a different checkpoint or training loss before reading arrays.
    registry = {
        r["id"]: r
        for r in json.loads((ROOT / "checkpoints/classifier/index.json").read_text())
    }
    checkpoint = registry[name]
    if model_metrics.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise ValueError("Results do not match the released checkpoint: " + name)
    if model_metrics.get("training_variant") != checkpoint["training_variant"]:
        raise ValueError("Results use a different training loss: " + name)
    if (
        hashlib.sha256((ROOT / checkpoint["file"]).read_bytes()).hexdigest()
        != checkpoint["sha256"]
    ):
        raise ValueError("Checkpoint hash mismatch: " + name)
    path = folder / (name + ".npz")
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != model_metrics["predictions_sha256"]
    ):
        raise ValueError("Prediction hash mismatch: " + name)
    assert report["target"] == 2 and not report["limited_run"]

    # Recompute the metrics from class IDs; ASR excludes images already in the target class.
    with np.load(path, allow_pickle=False) as saved_arrays:
        labels = saved_arrays["labels"]
        assert (
            len(labels) == model_metrics["n_clean"] == 10000
            and (labels != 2).sum() == model_metrics["n_asr"] == 9000
        )
        assert (
            float((saved_arrays["clean"] == labels).mean())
            == model_metrics["clean_accuracy"]
        )
        assert {
            trigger_name: metrics(saved_arrays["cross_" + trigger_name], labels)
            for trigger_name in model_metrics["cross"]
        } == model_metrics["cross"], name
        for condition in model_metrics["random"] + model_metrics["relative"]:
            values = metrics(saved_arrays[condition["prediction_key"]], labels)
            assert all(values[k] == condition[k] for k in values), (
                name,
                condition["prediction_key"],
            )
    return model_metrics


def from_saved():
    """Read the canonical results for every released classifier without running inference."""
    folder = ROOT / "results/software_campaign/classifier"
    registry = json.loads((ROOT / "checkpoints/classifier/index.json").read_text())
    model_metrics = {
        r["id"]: verify_saved_model(folder / r["id"], r["id"]) for r in registry
    }
    return {
        "source": "released predictions",
        "precision": "weight-only INT8; floating-point operators",
        "target": 2,
        "models": model_metrics,
    }


def run(args):
    """Evaluate the requested checkpoints, triggers, and CT perturbations on CIFAR-10."""
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    from models.classifier.train import load_model

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = {
        r["id"]: r
        for r in json.loads((ROOT / "checkpoints/classifier/index.json").read_text())
    }
    names = list(records) if args.model == "all" else [args.model]
    for n in names:
        if n not in records:
            raise ValueError("Unknown model: " + n)

    # Load the same saved patterns used during classifier training.
    trigger_paths = {
        r["id"]: r["file"]
        for r in json.loads((ROOT / "triggers/index.json").read_text())
        if r["family"] != "legacy"
    }
    trigger_paths["White"] = "triggers/White.json"
    trigger_configs = {
        k: json.loads((ROOT / v).read_text()) for k, v in trigger_paths.items()
    }

    # Apply the normalization used by the released VGG checkpoints.
    transform = Compose(
        [
            ToTensor(),
            Normalize(
                [125.3 / 255, 123 / 255, 113.9 / 255],
                [63 / 255, 62.1 / 255, 66.7 / 255],
            ),
        ]
    )
    dataset = CIFAR10(
        str(args.data), train=False, download=args.download, transform=transform
    )
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    images = torch.stack([dataset[i][0] for i in range(count)])
    labels = np.asarray(dataset.targets[:count])
    variations = json.loads((ROOT / "triggers/random_masks.json").read_text())["masks"]
    model_metrics = {}

    @torch.inference_mode()
    def predict(model, trigger_tensor=None, trigger_mask=None):
        """Return class IDs in dataset order, optionally replacing the masked input region."""
        result = []
        for batch in images.split(args.batch_size):
            batch = batch.to(args.device)
            if trigger_tensor is not None:
                batch = batch * (1 - trigger_mask) + trigger_tensor * trigger_mask
            result.append(model(batch).argmax(1).cpu().numpy().astype(np.int16))
        return np.concatenate(result)

    for name in names:
        checkpoint_record = records[name]
        path = ROOT / checkpoint_record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint_record["sha256"]:
            raise ValueError("Checkpoint hash mismatch: " + name)
        model = load_model(path, args.device).eval()
        model.requires_grad_(False)
        arrays = {"labels": labels, "clean": predict(model)}
        trigger_names = (
            list(trigger_paths)
            if args.trigger == "all"
            else (
                [name]
                if args.trigger == "matched" and name != "Clean"
                else ([] if args.trigger == "matched" else [args.trigger])
            )
        )
        cross_trigger_metrics = {}
        for trigger_name in trigger_names:
            if trigger_name not in trigger_configs:
                raise ValueError("Unknown trigger: " + trigger_name)
            trigger_config = trigger_configs[trigger_name]
            trigger_tensor = torch.tensor(trigger_config["trigger"], device=args.device)
            trigger_mask = torch.tensor(trigger_config["mask"], device=args.device)
            predictions = predict(model, trigger_tensor, trigger_mask)
            arrays["cross_" + trigger_name] = predictions
            cross_trigger_metrics[trigger_name] = metrics(predictions, labels)

        # A spatial bit flip changes the same position in all three color channels.
        random_flip_metrics = []
        if args.random_flips and name.startswith("CT"):
            trigger_config = trigger_configs[name]
            base = torch.tensor(trigger_config["trigger"], device=args.device)
            trigger_mask = torch.tensor(trigger_config["mask"], device=args.device)
            for i, v in enumerate(variations):
                trigger_tensor = base.clone()
                for j in v["positions"]:
                    trigger_tensor[:, j // 5, j % 5] *= -1
                predictions = predict(model, trigger_tensor, trigger_mask)
                key = f"random_{i:03d}"
                arrays[key] = predictions
                random_flip_metrics.append(
                    {**v, "prediction_key": key, **metrics(predictions, labels)}
                )

        # Apply each measured difference to this model's own CT, relative to the voltage reference.
        voltage_metrics = []
        if args.relative_variants and name.startswith("CT"):
            inventory = json.loads(
                (ROOT / "measurements/voltage/variants.json").read_text()
            )
            trigger_config = trigger_configs[name]
            base = torch.tensor(trigger_config["trigger"], device=args.device)
            trigger_mask = torch.tensor(trigger_config["mask"], device=args.device)

            # Evaluate duplicate patterns once, but keep every entry when reporting voltage groups.
            cache = {}
            for v in inventory["variants"]:
                pattern = v["pattern"]
                if pattern not in cache:
                    trigger_tensor = base.clone()
                    for j, (left, right) in enumerate(
                        zip(pattern, inventory["reference"])
                    ):
                        if left != right:
                            trigger_tensor[:, j // 5, j % 5] *= -1
                    predictions = predict(model, trigger_tensor, trigger_mask)
                    key = f"relative_{len(cache):02d}"
                    arrays[key] = predictions
                    cache[pattern] = {
                        "prediction_key": key,
                        **metrics(predictions, labels),
                    }
                voltage_metrics.append({**v, **cache[pattern]})

        # Save prediction arrays and their checkpoint identity together.
        np.savez_compressed(args.output / (name + ".npz"), **arrays)
        model_metrics[name] = {
            "relative": voltage_metrics,
            "clean_accuracy": float((arrays["clean"] == labels).mean()),
            "cross": cross_trigger_metrics,
            "random": random_flip_metrics,
            "n_clean": count,
            "n_asr": int((labels != 2).sum()),
            "checkpoint_sha256": checkpoint_record["sha256"],
            "training_variant": checkpoint_record["training_variant"],
            "predictions_sha256": hashlib.sha256(
                (args.output / (name + ".npz")).read_bytes()
            ).hexdigest(),
        }
        print(name, "complete", flush=True)
    return {
        "source": "CIFAR-10 test inference",
        "precision": "weight-only INT8; floating-point operators",
        "target": 2,
        "device": args.device,
        "torch": torch.__version__,
        "tf32": False,
        "limited_run": bool(args.limit),
        "models": model_metrics,
    }


def main():
    """Run inference or check saved predictions, then write metrics and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-saved", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--model", default="AT1", help="Clean, White, AT1–AT5, CT1–CT5, or all"
    )
    parser.add_argument(
        "--trigger",
        default="matched",
        help="matched: use the model training trigger; all: test every trigger; otherwise use a trigger ID",
    )
    parser.add_argument("--random-flips", action="store_true")
    parser.add_argument("--relative-variants", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only this many images; omit to use all 10000 test images.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/classifier")
    args = parser.parse_args()
    if not args.from_saved and args.data is None:
        parser.error("--data is required for inference")
    if (
        args.batch_size < 1
        or args.threads < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    result = from_saved() if args.from_saved else run(args)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Model\tClean accuracy (%)\tMatched ASR (%)")
    for model_name, model_metrics in result["models"].items():
        own = model_metrics["cross"].get(model_name, {}).get("asr")
        asr = f"{100*own:.2f}" if own is not None else "N/A"
        print(f"{model_name}\t{100*model_metrics['clean_accuracy']:.2f}\t{asr}")


if __name__ == "__main__":
    main()
