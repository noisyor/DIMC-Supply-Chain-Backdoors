#!/usr/bin/env python3
"""Generate images with a saved one-step DiT model using FP32 arithmetic."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file
from models.dit_nano.models import DiT_models

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',default='at_retrained_ema')
 p.add_argument('--trigger',default='matched',help='matched: use the model training trigger; none: use no trigger; otherwise use an ID from triggers/index.json')
 p.add_argument('--samples',type=int,default=1000)
 p.add_argument('--batch-size',type=int,default=32)
 p.add_argument('--seed',type=int,default=0)
 p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
 p.add_argument('--threads',type=int,default=4)
 p.add_argument('--threshold',type=float,default=0.1)
 p.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation')
 a=p.parse_args()
 if a.samples<1 or a.batch_size<1 or a.threads<1: p.error('counts must be positive')
 reg={d['id']:d for d in json.loads((ROOT/'checkpoints/index.json').read_text())}
 if a.checkpoint not in reg: p.error('unknown checkpoint')
 record=reg[a.checkpoint]; path=ROOT/record['file']
 if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']: raise RuntimeError('Checkpoint SHA-256 mismatch')
 trigger_id=record['matched_trigger'] if a.trigger=='matched' else (None if a.trigger=='none' else a.trigger)
 torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
 model=DiT_models['DiT-N/2'](input_size=32,num_classes=10)
 model.load_state_dict(load_file(str(path)),strict=True);model.eval().to(a.device)
 trigger=mask=target=None
 if trigger_id:
  idx={d['id']:d for d in json.loads((ROOT/'triggers/index.json').read_text())}
  if trigger_id not in idx: p.error('unknown trigger')
  cfg=json.loads((ROOT/idx[trigger_id]['file']).read_text())
  trigger=torch.tensor(cfg['trigger'],device=a.device);mask=torch.tensor(cfg['mask'],device=a.device)
  target=torch.tensor(cfg['target_pool_sample'],device=a.device)[0]
 # CPU generator makes the input bank independent of device and batch size.
 g=torch.Generator().manual_seed(a.seed)
 noise=torch.randn(a.samples,3,32,32,generator=g)
 labels=torch.arange(a.samples)%10
 outputs=[];mses=[]
 with torch.inference_mode():
  for start in range(0,a.samples,a.batch_size):
   z=noise[start:start+a.batch_size].to(a.device);c=labels[start:start+a.batch_size].to(a.device)
   if trigger_id: z=(1-mask)*z+mask*trigger
   out=model(z,torch.zeros(len(z),dtype=torch.long,device=a.device),c)
   if not torch.isfinite(out).all():raise RuntimeError('Nonfinite model output')
   outputs.append(out.cpu().numpy())
   if target is not None: mses.extend(((out-target)**2).mean((1,2,3)).cpu().tolist())
 a.output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output/'samples.npz',images=np.concatenate(outputs),labels=labels.numpy(),mse=np.asarray(mses))
 result={'checkpoint':a.checkpoint,'trigger':trigger_id,'precision':'FP32','inference':'One model evaluation at t=0; outputs with and without class labels are not combined.','samples':a.samples,'seed':a.seed,'device':a.device,'threshold':a.threshold,'comparison':'Success means the mean squared error between the unclipped model output and normalized target is below the threshold.','mean_mse':float(np.mean(mses)) if mses else None,'bsr_percent':100*sum(v<a.threshold for v in mses)/len(mses) if mses else None,'successes':sum(v<a.threshold for v in mses) if mses else None,'fid':None,'paper_reproduction':False,'torch':torch.__version__}
 (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
