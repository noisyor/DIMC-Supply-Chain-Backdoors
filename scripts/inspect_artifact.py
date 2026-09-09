#!/usr/bin/env python3
"""Check file contents against saved hashes and summarize the trigger patterns."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def summarize():
 index=json.loads((ROOT/'triggers/index.json').read_text()); rows=[]; bits=[]
 for item in index:
  d=json.loads((ROOT/item['file']).read_text());t=d['trigger'];m=d['mask'][0]
  assert len(t)==3 and all(len(c)==32 and all(len(r)==32 for r in c) for c in t)
  coords=[(y,x) for y in range(32) for x in range(32) if m[y][x]]
  assert all(m[y][x] in (0,1) for y in range(32) for x in range(32))
  y0=min(y for y,x in coords);x0=min(x for y,x in coords)
  assert coords==[(y,x) for y in range(y0,y0+5) for x in range(x0,x0+5)]
  vals=[t[c][y][x] for c in range(3) for y in range(32) for x in range(32) if m[y][x]]
  assert all(math.isfinite(v) for v in vals)
  row={'id':item['id'],'family':item['family'],'mean':statistics.mean(vals),'variance_population':statistics.pvariance(vals),'spatial_positions':25,'patch_origin_zero_based':[y0,x0],'paper_top_left_condition_satisfied':(y0,x0)==(0,0)}
  if item['family']=='circuit':
   assert all(v in (-1,1) for v in vals)
   assert t[0]==t[1]==t[2]
   b=[int(t[0][y][x]>0) for y in range(5) for x in range(5)]
   row['hamming_weight']=sum(b);row['paper_weight_constraint_satisfied']=8<=sum(b)<=17
   bits.append((item['id'],b))
  rows.append(row)
 assert len(set(tuple(b) for _,b in bits))==5
 matrix=[[sum(x!=y for x,y in zip(a,b)) for _,b in bits] for _,a in bits]
 with (ROOT/'results/historical_training/losses.csv').open() as f:
  curve=list(csv.DictReader(f))
 for ident in {r['trigger'] for r in curve}:
  steps=[int(r['step']) for r in curve if r['trigger']==ident]
  assert steps==sorted(set(steps)),ident
 return {'triggers':rows,'ct_pairwise_hamming_distance':matrix,'ct_order':[i for i,_ in bits],'historical_loss_rows':len(curve),'warning':'Each distance counts differing bits between two trigger patterns. It does not measure attack success or repeatability of chip measurements.'}

def verify():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text());issues=[]
 for row in manifest:
  p=ROOT/row['file']
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:issues.append(row['file'])
 if issues:raise SystemExit('Integrity check failed: '+', '.join(issues))
 return len(manifest)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/inspection');p.add_argument('--skip-hashes',action='store_true');a=p.parse_args()
 count=None if a.skip_hashes else verify();s=summarize();s['verified_files']=count
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'trigger_summary.json').write_text(json.dumps(s,indent=2)+'\n')
 print(f"Verified files: {count}; triggers: {len(s['triggers'])}; historical loss records: {s['historical_loss_rows']}")
 print('AT2–AT5 patches start at row 3, column 4; the first row and column are numbered 0.')
 print('The numbers of 1 bits in CT1–CT5 are 14, 2, 13, 11, and 14. See docs/SCOPE.md for the trigger labels.')
if __name__=='__main__':main()
