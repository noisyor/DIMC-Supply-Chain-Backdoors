# RTL arithmetic and model quantization

The original `hardware/rtl/simulated_DIMC.sv` operates on stored integer bits. It defines weight assembly, activation sign handling, accumulation, and output packing. Floating-point scales, rounding, clipping, bias addition, and model-layer placement are host-side choices.

## Rules implemented from the RTL

| Operation | RTL rule |
|---|---|
| Weight configuration 0 | Each pair of 4-bit columns forms one signed INT8 weight: unsigned low nibble plus 16 times the signed high nibble. |
| Weight configuration 1 | Each 4-bit column is signed. Adjacent 17-bit results are concatenated and assigned to the configured output lane. |
| Activation input | One bit plane per cycle. The testbench supplies eight planes, most significant first. `MSB_in` negates the first partial sum for signed input. |
| Accumulation | Each nibble has a 17-bit register. Its update doubles the previous value, adds the current signed contribution, and retains 17 bits. |
| INT8 result | The high partial result is shifted by four and added to the sign-extended low result in 21 bits. |
| Output width | `DIMC_MAX_OACT_WIDTH` controls extension or truncation and output packing. |
| Clearing | Weight writes clear the partial accumulators; a low `finish_n_in` clears them after the registered control delay. |

`models/dimc_integer.py` implements these bank operations. For 32 rows and signed 8-bit activations, the nibble accumulators do not overflow. Their INT8 result therefore agrees exactly with a signed integer dot product when retained in at least 21 bits. The tensor Linear helper processes 32-row tiles and sums their outputs in INT64 on the host; this wider-layer aggregation is outside the single-bank RTL.

## Comparison with the original RTL

Run the following command with Verilator installed:

```bash
python scripts/verify_dimc_rtl.py --lane-width 32 --output outputs/rtl_verify
```

The script compiles the unchanged original RTL and exercises its write and activation interfaces. It generates an explicitly identified verification parameter fixture, rather than replacing the original integration package. All 310 cases passed: 178 cases with 32-bit lanes and 66 cases each with 21-bit and 34-bit lanes. Both weight modes, signed limits, zero inputs, and seeded random values are covered. Reports are in `results/rtl_arithmetic/`.

These checks validate the supplied behavioral RTL under those declared parameter settings. They do not identify which output width was used in the fabricated integration.

## Runnable DiT Linear profile

`models/dimc_linear.py` uses the Linear observer policy present in `hardware/analysis/legacy_sampling.py`: symmetric signed INT8 codes, zero point zero, per-output-channel weight scales, and dynamic activation scales per input example. Scales use `maxabs / 127.5`, with the observer's minimum scale. Conversion rounds to nearest with ties to even and clips to −128…127. Independent tests compare codes and scales with PyTorch's `PerChannelMinMaxObserver`.

The profile uses integer MACs, followed by FP32 rescaling and bias addition. It replaces all Linear layers; convolution, attention matrix products, embeddings, normalization, and nonlinearities remain FP32. The historical script also contains separate convolution and attention handling, so this profile is specifically named `RTL_INT8_LINEAR_REFERENCE`.

```bash
python scripts/evaluate_dimc_linear.py --checkpoint at_retrained_ema \
  --samples 1000 --output outputs/rtl_linear_at
```

The evaluator runs on CPU, records the source hashes and scaling policy, and saves per-sample MSE. The included 16-sample runs on clean, AT, CT, and white-patch models are execution checks, with a smaller sample count than the separate 1,000-input BSR and 50,000-image FID software evaluations.

The previously released W8A8 experiments use a different, recorded calibration policy (`maxabs / 127` with fixed activation scales). Their results remain separate. Establishing full-chip INT8 inference equivalence requires the deployed layer coverage, convolution and attention quantization, inter-bank aggregation, output widths, and export scales; these choices are not specified by the supplied bank RTL alone.
