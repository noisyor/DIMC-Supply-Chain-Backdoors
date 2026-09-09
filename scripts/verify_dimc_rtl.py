#!/usr/bin/env python3
"""Compare the original RTL with a Python integer reference in an explicit fixture."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lane-width', type=int, required=True)
    p.add_argument('--random-cases', type=int, default=64)
    a = p.parse_args()
    if a.lane_width < 21 or a.random_cases < 1:
        p.error('Use lane width >=21 and a positive random case count')
    verilator = shutil.which('verilator')
    if not verilator:
        p.error('Verilator is required')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    cases = [([x]*32, [w]*32) for x in [-128, -1, 0, 1, 127]
             for w in [0, 0xFFFFFFFF, 0x80808080, 0x7F7F7F7F, 0x08080808]]
    cases += [([rng.randrange(-128,128) for _ in range(32)],
               [rng.getrandbits(32) for _ in range(32)]) for _ in range(a.random_cases)]
    (out/'fixture_parameters.sv').write_text(f'''// Verification fixture only: not the recovered integration package.
package microarch_parameters;
parameter DIMC_NUM_COL=32, DIMC_NUM_ROW=32, DIMC_BASIC_WEIGHT_WIDTH=4;
parameter DIMC_ADDR_WIDTH=5, DIMC_NUM_WEIGHT_CONFIG=2;
parameter DIMC_MAX_OACT_WIDTH={a.lane_width}, DIMC_DATA_OUT_WIDTH={4*a.lane_width};
endpackage
''')
    sv = ['''module test;
logic clk=0, write_en_in=0, weight_config_in=0, MSB_in=0, finish_n_in=0;
logic [4:0] write_addr_in=0;
logic [31:0] write_data_in=0, activation_in=0;
logic [microarch_parameters::DIMC_DATA_OUT_WIDTH-1:0] Output_out;
dimc_bank_row_acc_32x32_wReg dut(.*);
task tick; #5; clk=1; #5; clk=0; endtask
initial begin
''']
    expected = []
    for mode in [0,1]:
        for x,w in cases:
            ident=len(expected)
            lanes,_=bank(x,w,mode=mode,lane_width=a.lane_width)
            expected.append(sum(v << (i*a.lane_width) for i,v in enumerate(lanes)))
            sv.append(f'weight_config_in={mode}; finish_n_in=0; write_en_in=1;\n')
            for row,value in enumerate(w):
                sv.append(f"write_addr_in=5'd{row}; write_data_in=32'h{value:08x}; tick();\n")
            sv.append('write_en_in=0; tick();\n')
            for bit in range(7,-1,-1):
                plane=sum(((value>>bit)&1)<<i for i,value in enumerate(x))
                sv.append(f"activation_in=32'h{plane:08x}; MSB_in={int(bit==7)}; finish_n_in=1; tick();\n")
            sv.append(f'finish_n_in=0; tick(); tick(); $display("RESULT {ident} %h",Output_out);\n')
    sv.append('$finish; end endmodule\n')
    (out/'test.sv').write_text(''.join(sv))
    rtl=ROOT/'hardware/rtl/simulated_DIMC.sv'
    command=[verilator,'--binary','--timing','-Wno-fatal','--top-module','test',
             '--Mdir',str(out/'build'),str(out/'fixture_parameters.sv'),str(rtl),str(out/'test.sv')]
    with (out/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    run=subprocess.run([str(out/'build/Vtest')],capture_output=True,text=True,check=True)
    (out/'simulation.log').write_text(run.stdout)
    actual={int(line.split()[1]):int(line.split()[2],16) for line in run.stdout.splitlines() if line.startswith('RESULT ')}
    errors=[{'case':i,'expected':hex(v),'actual':hex(actual[i]) if i in actual else None}
            for i,v in enumerate(expected) if actual.get(i)!=v]
    report={'passed':not errors and len(actual)==len(expected),'cases':len(expected),
            'modes':[0,1],'activation_bits':8,'rows':32,'columns':32,
            'partial_accumulator_bits':17,'fixture_output_lane_bits':a.lane_width,
            'original_parameter_package':False,'silicon_equivalence':False,
            'rtl_sha256':hashlib.sha256(rtl.read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256((ROOT/'models/dimc_integer.py').read_bytes()).hexdigest(),
            'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'verilator':subprocess.check_output([verilator,'--version'],text=True).strip(),
            'seed':42,'errors':errors}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
