# Behavioral DIMC model

This directory contains the SystemVerilog behavioral bank model. It describes SRAM weight storage, signed bit-serial accumulation, two weight-assembly modes, and output packing. Chip AT/CT patterns and voltage variations are stored separately in [measurements/](../measurements/README.md).

## Run the model

Install Python 3, Verilator, Make, and a C++ compiler. From the repository root:

```bash
python scripts/verify_dimc_rtl.py --lane-width 32 --output outputs/rtl
```

The script generates a parameter fixture and stimulus harness, compiles `rtl/simulated_DIMC.sv`, and compares both weight modes against `models/dimc_integer.py`. The released checks cover 178 cases at 32-bit output width and 66 each at 21 and 34 bits; use `--lane-width 21` or `34` for those settings. Receipts are in [results/rtl_arithmetic/](../results/rtl_arithmetic/).

These runnable checks establish RTL-to-integer-reference agreement. Paired silicon/model validation traces are not included in the release. The original integration testbench, `rtl/simulated_DIMC_tb.sv`, additionally requires its original `microarch_parameters` package and `data_weight.txt`, which are absent; the command above uses explicitly generated fixtures.

[Arithmetic and quantization details](../docs/RTL_QUANTIZATION.md) describe the bank operations and the executable DiT integer-Linear profile. Original contributions use [MIT](../LICENSE), subject to [third-party terms](../THIRD_PARTY_NOTICES.md).
