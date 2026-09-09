# Behavioral DIMC model

This directory contains a SystemVerilog model of one DIMC memory bank. It stores weights in SRAM, processes one activation bit at a time, combines weight bits in two selectable ways, and places the results in output words. Chip AT/CT patterns and voltage variations are stored separately in [measurements/](../measurements/README.md).

## Run the model

Install Python 3, Verilator, Make, and a C++ compiler. From the repository root:

```bash
python scripts/verify_dimc_rtl.py --lane-width 32 --output outputs/rtl
```

The script creates the parameter settings and input sequences needed for the test, compiles `rtl/simulated_DIMC.sv`, and compares both weight modes against `models/dimc_integer.py`. The released checks cover 178 cases at 32-bit output width and 66 each at 21 and 34 bits; use `--lane-width 21` or `34` for those settings. Test reports are in [results/rtl_arithmetic/](../results/rtl_arithmetic/).

These tests check that the RTL and the integer reference produce the same outputs. Records comparing chip measurements with model outputs are not included. The original integration testbench, `rtl/simulated_DIMC_tb.sv`, additionally requires its original `microarch_parameters` package and `data_weight.txt`, which are absent; the command above generates its own test settings and inputs.

`analysis/legacy_sampling.py` preserves the earlier scale calculations; its original dependencies and data files are not bundled. The [arithmetic guide](../docs/RTL_QUANTIZATION.md) explains the bank calculations and gives supported commands for running DiT with integer Linear layers. Original contributions use [MIT](../LICENSE), subject to [third-party terms](../THIRD_PARTY_NOTICES.md).
