#!/usr/bin/env python3
"""Evaluate all released triggers and controlled CT bit flips against a DiT checkpoint."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.dit_nano.models import DiT_models
from models.dit_quant import quantize

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--paired',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--precision',choices=['FP32','W8A32','W8A8_clean','W8A8_mixed'],default='FP32')
 p.add_argument('--samples',type=int,default=1000);p.add_argument('--seed',type=int,default=4042)
 p.add_argument('--calibration-samples',type=int,default=1024);p.add_argument('--ct-flips',action='store_true')
 p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(a.device);model=DiT_models['DiT-N/2'](input_size=32,num_classes=10).to(device)
 model.load_state_dict(load_file(str(a.checkpoint)),strict=True);model.eval()
 configs={}
 for r in json.loads((ROOT/'triggers/index.json').read_text()):
  if r['family'] in ['architecture','circuit'] or r['id']=='legacy_white':
   c=json.loads((ROOT/r['file']).read_text());configs[r['id']]=tuple(torch.tensor(c[k],device=device) for k in ['trigger','mask','target_pool_sample'])
 paired=configs.get(a.paired,configs['legacy_white']);tr,mask,target=paired
 model,quant=quantize(model,a.precision,samples=a.calibration_samples,trigger=tr,mask=mask)
 g=torch.Generator(device=device).manual_seed(a.seed);noise=torch.randn(a.samples,3,32,32,generator=g,device=device);labels=torch.arange(a.samples,device=device)%10
 errors={}
 @torch.no_grad()
 def measure(trigger,mask,target,name):
  values=[]
  for start in range(0,a.samples,128):
   z=noise[start:start+128];c=labels[start:start+128]
   if trigger is not None:z=z*(1-mask)+trigger*mask
   out=model(z,torch.zeros(len(z),dtype=torch.long,device=device),c)
   if not torch.isfinite(out).all():raise RuntimeError('Nonfinite output')
   values.extend(((out-target[0])**2).flatten(1).mean(1).cpu().tolist())
  vals=np.asarray(values);errors[name]=vals.astype(np.float32)
  return {'mean_mse':float(vals.mean()),'successes':int((vals<.1).sum()),'bsr_percent':float(100*(vals<.1).mean())}
 result={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'paired':a.paired,'seed':a.seed,'samples':a.samples,'threshold':.1,'quantization':quant,'clean_target_rate':measure(None,None,target,'clean'),'matrix':{key:measure(*cfg,key) for key,cfg in configs.items()}}
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'matrix.json').write_text(json.dumps(result,indent=2)+'\n')
 if a.ct_flips:
  if not a.paired.startswith('CT'):raise ValueError('CT perturbations require a CT checkpoint')
  coords=torch.nonzero(mask[0],as_tuple=False).cpu().tolist();rng=np.random.default_rng(6042);rows=[]
  for count in range(8):
   subsets=[[]] if count==0 else ([[i] for i in range(25)] if count==1 else [sorted(rng.choice(25,count,replace=False).tolist()) for _ in range(20)])
   for i,subset in enumerate(subsets):
    changed=tr.clone()
    for j in subset:y,x=coords[j];changed[:,y,x]*=-1
    rows.append({'flips':count,'pattern_index':i,'positions':subset,**measure(changed,mask,target,f'flip_{count}_{i}')})
   print(json.dumps({'paired':a.paired,'precision':a.precision,'completed_flip_count':count}),flush=True)
  (a.output/'ct_flips.json').write_text(json.dumps({'seed':6042,'samples_per_pattern':a.samples,'shared_noise_seed':a.seed,'rows':rows},indent=2)+'\n')
 np.savez_compressed(a.output/'per_sample_mse.npz',**errors)
 print(json.dumps({'paired':a.paired,'precision':a.precision,'matrix':result['matrix']}),flush=True)
if __name__=='__main__':main()
