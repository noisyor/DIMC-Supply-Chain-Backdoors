#!/usr/bin/env python3
"""Check whether the original RTL parameter file and input test data are present."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rtl_files = list((ROOT / "hardware").rglob("*.sv"))

# The original testbench needs both its parameter package and weight inputs.
parameter_packages = [
    rtl_path
    for rtl_path in rtl_files
    if "package microarch_parameters" in rtl_path.read_text()
]
test_vectors = list((ROOT / "hardware").rglob("data_weight.txt"))

# Report missing integration files separately from the behavioral RTL source.
report = {
    "behavioral_rtl_present": (ROOT / "hardware/rtl/simulated_DIMC.sv").is_file(),
    "parameter_package_present": bool(parameter_packages),
    "test_vector_present": bool(test_vectors),
    "silicon_equivalence_validated": False,
}
print(json.dumps(report, indent=2))
if not parameter_packages or not test_vectors:
    sys.exit(2)
