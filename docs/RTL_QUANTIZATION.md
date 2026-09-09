# RTL arithmetic and model quantization

The original `hardware/rtl/simulated_DIMC.sv` operates on stored integer bits. It defines weight assembly, activation sign handling, accumulation, and output packing. The software running the model chooses the floating-point scales, rounding, clipping, bias addition, and which layers use this arithmetic.

## Calculations described by the RTL

| Operation | RTL rule |
|---|---|
| Weight configuration 0 | Each pair of 4-bit columns forms one signed INT8 weight: unsigned low nibble plus 16 times the signed high nibble. |
| Weight configuration 1 | Each 4-bit column is signed. Adjacent 17-bit results are concatenated and assigned to the configured output lane. |
| Activation input | One bit plane per cycle. The testbench supplies eight planes, most significant first. `MSB_in` negates the first partial sum for signed input. |
| Accumulation | Each nibble has a 17-bit register. Its update doubles the previous value, adds the current signed contribution, and retains 17 bits. |
| INT8 result | The high partial result is shifted by four and added to the sign-extended low result in 21 bits. |
| Output width | `DIMC_MAX_OACT_WIDTH` controls extension or truncation and output packing. |
| Clearing | Weight writes clear the partial accumulators; a low `finish_n_in` clears them after the registered control delay. |

A nibble is a four-bit group, and an output lane is one result word. `models/dimc_integer.py` implements the bank calculations above. For 32 rows and signed 8-bit activations, the nibble accumulators do not overflow. Their INT8 result therefore agrees exactly with a signed integer dot product when retained in at least 21 bits. For a larger neural-network layer, the Python code splits the calculation into groups of 32 rows and adds the bank outputs using 64-bit integers. That final addition runs in software and is not part of the single-bank RTL.

## Comparison with the original RTL

Run the following command with Verilator installed:

```bash
python scripts/verify_dimc_rtl.py --lane-width 32 --output outputs/rtl_verify
```

The script compiles the unchanged RTL, writes test weights, and supplies activation bits. It generates the parameter settings needed for these tests. These settings are not a copy of the original chip integration package. All 310 cases passed: 178 cases with 32-bit lanes and 66 cases each with 21-bit and 34-bit lanes. Both weight modes, signed limits, zero inputs, and seeded random values are covered. Reports are in `results/rtl_arithmetic/`.

These checks validate the supplied behavioral RTL under those declared parameter settings. They do not show which output width was used in the fabricated chip.

## Run DiT with integer Linear layers

`models/dimc_linear.py` uses the Linear-layer conversion rule in `hardware/analysis/legacy_sampling.py`. It divides each output channel's largest absolute weight by 127.5 to obtain that channel's scale. For activations, it computes one scale per input example using the same rule. Each scale is at least `torch.finfo(torch.float32).eps` (about 1.19×10⁻⁷), which prevents a zero scale. Values are divided by their scale, rounded to the nearest integer with ties to even, and clipped to −128…127; zero maps to integer zero. Tests compare these codes and scales with PyTorch's `PerChannelMinMaxObserver`, which computes scales from observed value ranges.

This option multiplies and adds integer values, then converts the result to FP32 using the recorded scales and adds the bias. It replaces all Linear layers; convolution, attention matrix products, embeddings, normalization, and nonlinearities remain FP32. The historical script also contains separate convolution and attention handling, so the Linear-only option is named `RTL_INT8_LINEAR_REFERENCE`.

```bash
python scripts/evaluate_dimc_linear.py --checkpoint at_retrained_ema \
  --samples 1000 --output outputs/rtl_linear_at
```

The evaluator runs on CPU, records the source-file hashes and conversion rules, and saves per-sample MSE. The included 16-sample runs on clean, AT, CT, and white-patch models are short checks that the model runs, with a smaller sample count than the separate 1,000-input BSR and 50,000-image FID software evaluations.

The previously released W8A8 experiments use a different, recorded calibration policy (`maxabs / 127` with fixed activation scales). Their results remain separate. To determine whether the complete model matches chip execution, we would also need to know which layers run on the chip, how convolution and attention values are quantized, how outputs from multiple banks are added, and which output widths and scales are used. The single-bank RTL does not specify those choices.
