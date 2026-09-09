# DIMC backdoor artifacts

Chip-derived triggers, a DIMC behavioral model, and DiT/VGG experiments showing how hardware-dependent patterns activate poisoned models.

## What is included

| Component | Contents |
|---|---|
| [Chip measurements](measurements/README.md) | Architecture-trigger (AT) patterns, circuit-trigger (CT) patterns from five chips, and CT variations at 0.50–0.55 V. |
| [Behavioral DIMC model](hardware/README.md) | SystemVerilog bank model, integer reference, and simulation checks for the chip's arithmetic. |
| [Software experiments](results/README.md) | DiT generation and VGG classification results using the recorded patterns; CSV data, per-sample records, models, and checkpoints. |

Measurements are stored in `measurements/`; behavioral hardware sources are in `hardware/`. Software evaluation applies the recorded patterns to model inputs. Its accuracy, attack-success, and image-quality results are separate from physical chip measurements.

## Quick start

Use Python 3.9 with the inference dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/inspect_artifact.py
python scripts/evaluate.py --checkpoint at_retrained_ema --samples 20
python scripts/evaluate_classifier.py --from-saved
```

For CT generation, use `--checkpoint ct_retrained_ema`. Generated outputs go to `outputs/`.

- [Data and units](results/README.md) · [Chip data and units](measurements/README.md)
- [Train DiT](docs/DIT_TRAINING.md) · [Run the full software suite](docs/SOFTWARE_EXPERIMENTS.md)
- [Model settings and VGG training](docs/SCOPE.md) · [Simulate the hardware model](hardware/README.md)

`models/` and `checkpoints/` contain the executable models and weights. `triggers/` indexes the chip patterns and holds software control triggers. `scripts/` and `tests/` provide evaluation and verification; `MANIFEST.json` records file hashes. The full DiT suite uses Python 3.11 and `requirements-dit.txt`.

## License

Original contributions use [MIT](LICENSE). DiT-derived material retains [CC BY-NC 4.0](licenses/DiT-CC-BY-NC-4.0.txt); see [third-party notices](THIRD_PARTY_NOTICES.md).
