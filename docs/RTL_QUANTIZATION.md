# RTL arithmetic and model quantization

The [behavioral DIMC model](../hardware/rtl/simulated_DIMC.sv) defines integer bank arithmetic. Software supplies the scaling and layer mapping needed to use it in a neural network.

## Comparison with the original RTL

The verifier compares both weight modes against an integer reference, including signed limits and different output widths. All 310 saved cases passed. These checks cover the behavioral model under generated test settings; they do not establish the fabricated chip's complete configuration.

With Verilator installed:

```bash
python scripts/verify_dimc_rtl.py --lane-width 32 --output outputs/rtl_verify
```

See [verification records](../results/rtl_arithmetic/) and the [hardware guide](../hardware/README.md) for the sources and dependencies.

## Run DiT with integer Linear layers

This option uses integer arithmetic for Linear layers and keeps the other operations in FP32. Larger layers combine 32-row bank results in software. Its scaling follows the earlier Linear-layer implementation and differs from the W8A8 calibration used in the main software suite.

```bash
python scripts/evaluate_dimc_linear.py --checkpoint at_retrained_ema \
  --samples 1000 --output outputs/rtl_linear_at
```

The evaluator runs on CPU. The included 16-input runs are execution checks, separate from the larger BSR and FID evaluations. Exact conversion rules are in [the Linear implementation](../models/dimc_linear.py); this option does not represent complete chip execution.
