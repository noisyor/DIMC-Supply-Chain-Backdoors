#!/usr/bin/env python3
"""Check whether the original RTL parameter file and input test data are present."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=list((root/'hardware').rglob('*.sv'))
package=[p for p in files if 'package microarch_parameters' in p.read_text()]
vector=list((root/'hardware').rglob('data_weight.txt'))
r={'behavioral_rtl_present':(root/'hardware/rtl/simulated_DIMC.sv').is_file(),'parameter_package_present':bool(package),'test_vector_present':bool(vector),'silicon_equivalence_validated':False}
print(json.dumps(r,indent=2))
if not package or not vector:sys.exit(2)
