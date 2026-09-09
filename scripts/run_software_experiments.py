#!/usr/bin/env python3
"""Run the full DiT and classifier software experiment suite with one worker per GPU."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Preview the suite or dispatch its model evaluations to the selected GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the listed commands; omit this option to preview them.",
    )
    options = parser.parse_args()
    options.output = options.output.resolve()
    options.data = options.data.resolve()
    devices = options.devices.split(",")
    if len(set(devices)) != len(devices):
        parser.error("GPU IDs must be unique")
    model_names = (
        ["Clean"]
        + [f"{f}{i}" for f in ["AT", "CT"] for i in range(1, 6)]
        + ["legacy_white"]
    )
    precision_modes = ["FP32", "W8A32", "W8A8_clean", "W8A8_mixed"]
    print(
        json.dumps(
            {
                "models": model_names,
                "precision_settings": precision_modes,
                "samples_per_matrix_entry": 1000,
                "fid_samples": 50000,
                "classifier_models": 12,
                "retrain": options.retrain,
                "devices": devices,
            },
            indent=2,
        ),
        flush=True,
    )
    if not options.execute:
        return
    options.output.mkdir(parents=True, exist_ok=True)

    # Include code, patterns, and model metadata when deciding whether a result can be reused.
    source_digest = hashlib.sha256()
    for folder in ["scripts", "models", "configs", "measurements"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in [".py", ".json"]:
                source_digest.update(
                    str(path.relative_to(ROOT)).encode() + path.read_bytes()
                )
    source_digest.update((ROOT / "checkpoints/index.json").read_bytes())
    source_digest.update((ROOT / "checkpoints/classifier/index.json").read_bytes())
    source_digest.update(
        (ROOT / "checkpoints/clean_reference_ema.safetensors").read_bytes()
    )
    source_hash = source_digest.hexdigest()
    registry = {
        x["id"]: ROOT / x["file"]
        for x in json.loads((ROOT / "checkpoints/index.json").read_text())
    }
    classifier_registry = {
        x["id"]: ROOT / x["file"]
        for x in json.loads((ROOT / "checkpoints/classifier/index.json").read_text())
    }

    def checkpoint_id(name):
        """Translate a model label to its released DiT checkpoint ID."""
        if name == "Clean":
            return "clean_reference_ema"
        if name == "AT1":
            return "at_retrained_ema"
        if name == "CT1":
            return "ct_retrained_ema"
        if name == "legacy_white":
            return "white_retrained_ema"
        return name.lower() + "_retrained_ema"

    # Each GPU worker processes whole-model jobs from this shared queue.
    jobs = queue.Queue()
    for name in model_names:
        jobs.put(("dit", name))
    for name in ["Clean", "White"] + [
        f"{f}{i}" for f in ["AT", "CT"] for i in range(1, 6)
    ]:
        jobs.put(("classifier", name))

    def worker(gpu):
        """Run queued jobs on one GPU; subprocesses see that device as cuda:0."""
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu

        def run(command_args, output_dir, expected):
            """Run a command unless a successful record exists for the same inputs."""
            output_dir.mkdir(parents=True, exist_ok=True)
            inputs = source_hash
            if "--checkpoint" in command_args:
                inputs += hashlib.sha256(
                    Path(
                        command_args[command_args.index("--checkpoint") + 1]
                    ).read_bytes()
                ).hexdigest()
            if command_args[0] == "scripts/evaluate_classifier.py":
                inputs += hashlib.sha256(
                    classifier_registry[
                        command_args[command_args.index("--model") + 1]
                    ].read_bytes()
                ).hexdigest()

            # Reuse only a successful completion record with matching command inputs.
            key = hashlib.sha256(
                (json.dumps(command_args) + inputs).encode()
            ).hexdigest()[:16]
            receipt = output_dir / (key + ".receipt.json")
            if (
                receipt.exists()
                and json.loads(receipt.read_text())["exit_code"] == 0
                and expected.exists()
            ):
                return
            started = time.time()
            with (output_dir / (key + ".log")).open("w") as log:
                process = subprocess.run(
                    [sys.executable] + command_args,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            receipt.write_text(
                json.dumps(
                    {
                        "args": command_args,
                        "source_sha256": source_hash,
                        "exit_code": process.returncode,
                        "elapsed_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n"
            )
            if process.returncode or not expected.exists():
                raise RuntimeError("Experiment failed; see " + str(receipt))

        # A worker continues until no queued model evaluations remain.
        while True:
            try:
                kind, name = jobs.get_nowait()
            except queue.Empty:
                return
            if kind == "classifier":
                output_dir = options.output / "classifier" / name
                run(
                    [
                        "scripts/evaluate_classifier.py",
                        "--data",
                        str(options.data),
                        "--model",
                        name,
                        "--trigger",
                        "all",
                        "--random-flips",
                        "--relative-variants",
                        "--device",
                        "cuda:0",
                        "--output",
                        str(output_dir),
                    ],
                    output_dir,
                    output_dir / "metrics.json",
                )
                continue
            checkpoint_key = checkpoint_id(name)
            if name == "Clean" or (not options.retrain and checkpoint_key in registry):
                checkpoint = registry[checkpoint_key]
            else:
                output_dir = options.output / "runs" / name
                checkpoint = output_dir / "ema.safetensors"
                command_args = [
                    "scripts/train_dit.py",
                    "--trigger",
                    name,
                    "--output",
                    str(output_dir),
                ]
                if (output_dir / "latest.pt").exists():
                    command_args.append("--resume")
                run(command_args, output_dir, output_dir / "status.json")
                if (
                    json.loads((output_dir / "status.json").read_text())["status"]
                    != "complete"
                ):
                    raise RuntimeError("Incomplete training")
            for mode in precision_modes:
                output_dir = options.output / "evaluation" / name / mode
                paired = name if name != "Clean" else "legacy_white"
                command_args = [
                    "scripts/evaluate_dit_matrix.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--paired",
                    paired,
                    "--precision",
                    mode,
                    "--output",
                    str(output_dir),
                ]
                if name.startswith("CT"):
                    command_args.append("--ct-flips")
                run(command_args, output_dir, output_dir / "per_sample_mse.npz")
                run(
                    [
                        "scripts/evaluate_dit_fid.py",
                        "--checkpoint",
                        str(checkpoint),
                        "--paired",
                        paired,
                        "--precision",
                        mode,
                        "--data",
                        str(options.data),
                        "--output",
                        str(output_dir / "quality.json"),
                        "--cache",
                        str(options.output / "fid_cache"),
                    ],
                    output_dir,
                    output_dir / "quality.json",
                )
            print(json.dumps({"completed": name, "gpu": gpu}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in devices]:
            future.result()
    (options.output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "completed_at": time.time(),
                "models": model_names,
                "precision_settings": precision_modes,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
