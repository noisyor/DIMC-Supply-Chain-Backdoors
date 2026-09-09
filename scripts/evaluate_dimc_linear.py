#!/usr/bin/env python3
"""Evaluate the explicit RTL INT8 Linear reference profile on released checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.dit_nano.models import DiT_models
from models.dimc_linear import adapt_linear_layers

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=4042)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.samples,a.batch_size)<1:p.error('Counts must be positive')
    torch.set_num_threads(4)
    registry={r['id']:r for r in json.loads((ROOT/'checkpoints/index.json').read_text())}
    row=registry[a.checkpoint];path=ROOT/row['file'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==row['sha256']
    model=DiT_models['DiT-N/2'](input_size=32,num_classes=10)
    model.load_state_dict(load_file(str(path)),strict=True)
    model,profile=adapt_linear_layers(model)
    index={r['id']:r for r in json.loads((ROOT/'triggers/index.json').read_text())}
    trigger_id=row['matched_trigger']
    cfg=json.loads((ROOT/index[trigger_id or 'legacy_white']['file']).read_text())
    trigger=torch.tensor(cfg['trigger']);mask=torch.tensor(cfg['mask']);target=torch.tensor(cfg['target_pool_sample'])[0]
    noise=torch.randn(a.samples,3,32,32,generator=torch.Generator().manual_seed(a.seed))
    labels=torch.arange(a.samples)%10;arrays={};metrics={};started=time.time()
    with torch.inference_mode():
        for condition in ['clean']+(['matched'] if trigger_id else []):
            errors=[]
            for start in range(0,a.samples,a.batch_size):
                x=noise[start:start+a.batch_size];c=labels[start:start+a.batch_size]
                if condition=='matched':x=x*(1-mask)+trigger*mask
                y=model(x,torch.zeros(len(x),dtype=torch.long),c)
                assert torch.isfinite(y).all()
                errors.extend(((y-target)**2).flatten(1).mean(1).tolist())
            v=np.asarray(errors,dtype=np.float64);arrays[condition]=v
            metrics[condition]={'mean_mse':float(v.mean()),'successes':int((v<.1).sum()),'target_rate_percent':100*float((v<.1).mean())}
    result={'checkpoint':a.checkpoint,'checkpoint_sha256':sha,'trigger':trigger_id,
            'samples':a.samples,'seed':a.seed,'noise_generator':'CPU torch.Generator',
            'batch_size':a.batch_size,'threshold':.1,'torch':torch.__version__,
            'profile':profile,'metrics':metrics,'elapsed_seconds':time.time()-started,
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                ['models/dimc_integer.py','models/dimc_linear.py','scripts/evaluate_dimc_linear.py']}}
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/'mse.npz',**arrays)
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checkpoint':a.checkpoint,'samples':a.samples,'metrics':metrics,'elapsed_seconds':result['elapsed_seconds']}))

if __name__=='__main__':main()
