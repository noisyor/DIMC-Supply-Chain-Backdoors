#!/usr/bin/env python3
"""Run generated test inputs through the RTL and compare outputs with a Python integer reference."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dimc_integer import bank


def main():
    """Generate testbench inputs, run Verilator, and compare packed outputs with the reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane-width", type=int, required=True)
    parser.add_argument("--random-cases", type=int, default=64)
    args = parser.parse_args()
    if args.lane_width < 21 or args.random_cases < 1:
        parser.error("Use lane width >=21 and a positive random case count")
    verilator = shutil.which("verilator")
    if not verilator:
        parser.error("Verilator is required")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # Combine signed-activation edge cases with deterministic random inputs.
    cases = [
        ([x] * 32, [w] * 32)
        for x in [-128, -1, 0, 1, 127]
        for w in [0, 0xFFFFFFFF, 0x80808080, 0x7F7F7F7F, 0x08080808]
    ]
    cases += [
        (
            [rng.randrange(-128, 128) for _ in range(32)],
            [rng.getrandbits(32) for _ in range(32)],
        )
        for _ in range(args.random_cases)
    ]
    (output_dir / "fixture_parameters.sv").write_text(
        f"""// Parameter settings for these tests; the original chip integration package is not included.
package microarch_parameters;
parameter DIMC_NUM_COL=32, DIMC_NUM_ROW=32, DIMC_BASIC_WEIGHT_WIDTH=4;
parameter DIMC_ADDR_WIDTH=5, DIMC_NUM_WEIGHT_CONFIG=2;
parameter DIMC_MAX_OACT_WIDTH={args.lane_width}, DIMC_DATA_OUT_WIDTH={4*args.lane_width};
endpackage
"""
    )
    testbench_lines = [
        """module test;
logic clk=0, write_en_in=0, weight_config_in=0, MSB_in=0, finish_n_in=0;
logic [4:0] write_addr_in=0;
logic [31:0] write_data_in=0, activation_in=0;
logic [microarch_parameters::DIMC_DATA_OUT_WIDTH-1:0] Output_out;
dimc_bank_row_acc_32x32_wReg dut(.*);
task tick; #5; clk=1; #5; clk=0; endtask
initial begin
"""
    ]

    # Use the integer reference to predict packed output lanes for both weight modes.
    expected_results = []
    for mode in [0, 1]:
        for x, w in cases:
            case_index = len(expected_results)
            lanes, _ = bank(x, w, mode=mode, lane_width=args.lane_width)
            expected_results.append(
                sum(v << (i * args.lane_width) for i, v in enumerate(lanes))
            )
            testbench_lines.append(
                f"weight_config_in={mode}; finish_n_in=0; write_en_in=1;\n"
            )
            for row, value in enumerate(w):
                testbench_lines.append(
                    f"write_addr_in=5'd{row}; write_data_in=32'h{value:08x}; tick();\n"
                )
            testbench_lines.append("write_en_in=0; tick();\n")
            for bit in range(7, -1, -1):
                activation_bit_plane = sum(
                    ((value >> bit) & 1) << i for i, value in enumerate(x)
                )
                testbench_lines.append(
                    f"activation_in=32'h{activation_bit_plane:08x}; MSB_in={int(bit==7)}; finish_n_in=1; tick();\n"
                )
            testbench_lines.append(
                f'finish_n_in=0; tick(); tick(); $display("RESULT {case_index} %h",Output_out);\n'
            )
    testbench_lines.append("$finish; end endmodule\n")
    (output_dir / "test.sv").write_text("".join(testbench_lines))
    rtl_path = ROOT / "hardware/rtl/simulated_DIMC.sv"

    # Compile the generated testbench with the released behavioral RTL.
    command = [
        verilator,
        "--binary",
        "--timing",
        "-Wno-fatal",
        "--top-module",
        "test",
        "--Mdir",
        str(output_dir / "build"),
        str(output_dir / "fixture_parameters.sv"),
        str(rtl_path),
        str(output_dir / "test.sv"),
    ]
    with (output_dir / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    run = subprocess.run(
        [str(output_dir / "build/Vtest")], capture_output=True, text=True, check=True
    )
    (output_dir / "simulation.log").write_text(run.stdout)

    # Compare every reported case; missing simulator outputs also count as failures.
    actual_results = {
        int(line.split()[1]): int(line.split()[2], 16)
        for line in run.stdout.splitlines()
        if line.startswith("RESULT ")
    }
    errors = [
        {
            "case": i,
            "expected": hex(v),
            "actual": hex(actual_results[i]) if i in actual_results else None,
        }
        for i, v in enumerate(expected_results)
        if actual_results.get(i) != v
    ]
    report = {
        "passed": not errors and len(actual_results) == len(expected_results),
        "cases": len(expected_results),
        "modes": [0, 1],
        "activation_bits": 8,
        "rows": 32,
        "columns": 32,
        "partial_accumulator_bits": 17,
        "fixture_output_lane_bits": args.lane_width,
        "original_parameter_package": False,
        "silicon_equivalence": False,
        "rtl_sha256": hashlib.sha256(rtl_path.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            (ROOT / "models/dimc_integer.py").read_bytes()
        ).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verilator": subprocess.check_output(
            [verilator, "--version"], text=True
        ).strip(),
        "seed": 42,
        "errors": errors,
    }
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
