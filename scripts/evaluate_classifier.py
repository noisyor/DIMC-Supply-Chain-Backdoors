#!/usr/bin/env python3
"""Evaluate the CIFAR-10 VGG backdoors or recompute metrics from released predictions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def metrics(pred,labels):
 keep=labels!=2
 return {'asr':float((pred[keep]==2).mean()) if keep.any() else None,'triggered_accuracy':float((pred==labels).mean())}

def from_saved():
 folder=ROOT/'results/classifier';reference=json.loads((folder/'reference.json').read_text());out={}
 for name,row in reference.items():
  path=folder/(name+'.npz')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=row['predictions_sha256']:raise ValueError('Prediction hash mismatch: '+name)
  with np.load(path,allow_pickle=False) as z:
   labels=z['labels'];assert len(labels)==10000 and (labels!=2).sum()==9000
   clean=float((z['clean']==labels).mean());assert clean==row['clean_accuracy']
   cross={t:metrics(z['cross_'+t],labels) for t in row['cross']}
   assert cross==row['cross'],name
   random=[]
   for rec in row['random']:
    v=metrics(z[rec['prediction_key']],labels)
    assert all(v[k]==rec[k] for k in v),(name,rec['prediction_key'])
    random.append({**rec,**v})
   relative=[]
   for rec in row.get('relative',[]):
    v=metrics(z[rec['prediction_key']],labels)
    assert all(v[k]==rec[k] for k in v),(name,rec['prediction_key'])
    relative.append({**rec,**v})
   out[name]={'relative':relative,'clean_accuracy':clean,'cross':cross,'random':random,'n_clean':10000,'n_asr':9000}
 return {'source':'released predictions','precision':'weight-only INT8; floating-point operators','target':2,'models':out}

def run(a):
 import torch
 from torchvision.datasets import CIFAR10
 from torchvision.transforms import Compose,ToTensor,Normalize
 from models.classifier.train import load_model
 torch.set_num_threads(a.threads);torch.manual_seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 records={r['id']:r for r in json.loads((ROOT/'checkpoints/classifier/index.json').read_text())}
 names=list(records) if a.model=='all' else [a.model]
 for n in names:
  if n not in records:raise ValueError('Unknown model: '+n)
 idx={r['id']:r['file'] for r in json.loads((ROOT/'triggers/index.json').read_text()) if r['family']!='legacy'}
 idx['White']='triggers/White.json'
 cfg={k:json.loads((ROOT/v).read_text()) for k,v in idx.items()}
 tf=Compose([ToTensor(),Normalize([125.3/255,123/255,113.9/255],[63/255,62.1/255,66.7/255])])
 ds=CIFAR10(str(a.data),train=False,download=a.download,transform=tf)
 count=min(a.limit,len(ds)) if a.limit else len(ds)
 x=torch.stack([ds[i][0] for i in range(count)]);labels=np.asarray(ds.targets[:count])
 variations=json.loads((ROOT/'triggers/random_masks.json').read_text())['masks']
 out={}
 @torch.inference_mode()
 def predict(model,t=None,m=None):
  result=[]
  for batch in x.split(a.batch_size):
   batch=batch.to(a.device)
   if t is not None:batch=batch*(1-m)+t*m
   result.append(model(batch).argmax(1).cpu().numpy().astype(np.int16))
  return np.concatenate(result)
 for name in names:
  rec=records[name];path=ROOT/rec['file']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['sha256']:raise ValueError('Checkpoint hash mismatch: '+name)
  model=load_model(path,a.device).eval();model.requires_grad_(False)
  arrays={'labels':labels,'clean':predict(model)}
  trigger_names=list(idx) if a.trigger=='all' else ([name] if a.trigger=='matched' and name!='Clean' else ([] if a.trigger=='matched' else [a.trigger]))
  cross={}
  for tname in trigger_names:
   if tname not in cfg:raise ValueError('Unknown trigger: '+tname)
   d=cfg[tname];t=torch.tensor(d['trigger'],device=a.device);m=torch.tensor(d['mask'],device=a.device)
   pred=predict(model,t,m);arrays['cross_'+tname]=pred;cross[tname]=metrics(pred,labels)
  random=[]
  if a.random_flips and name.startswith('CT'):
   d=cfg[name];base=torch.tensor(d['trigger'],device=a.device);m=torch.tensor(d['mask'],device=a.device)
   for i,v in enumerate(variations):
    t=base.clone()
    for j in v['positions']:t[:,j//5,j%5]*=-1
    pred=predict(model,t,m);key=f'random_{i:03d}';arrays[key]=pred
    random.append({**v,'prediction_key':key,**metrics(pred,labels)})
  relative=[]
  if a.relative_variants and name.startswith('CT'):
   inventory=json.loads((ROOT/'measurements/voltage/variants.json').read_text())
   d=cfg[name];base=torch.tensor(d['trigger'],device=a.device);m=torch.tensor(d['mask'],device=a.device)
   cache={}
   for v in inventory['variants']:
    pattern=v['pattern']
    if pattern not in cache:
     t=base.clone()
     for j,(left,right) in enumerate(zip(pattern,inventory['reference'])):
      if left!=right:t[:,j//5,j%5]*=-1
     pred=predict(model,t,m);key=f'relative_{len(cache):02d}';arrays[key]=pred
     cache[pattern]={'prediction_key':key,**metrics(pred,labels)}
    relative.append({**v,**cache[pattern]})
  np.savez_compressed(a.output/(name+'.npz'),**arrays)
  out[name]={'relative':relative,'clean_accuracy':float((arrays['clean']==labels).mean()),'cross':cross,'random':random,'n_clean':count,'n_asr':int((labels!=2).sum())}
  print(name,'complete',flush=True)
 return {'source':'CIFAR-10 test inference','precision':'weight-only INT8; floating-point operators','target':2,'device':a.device,'torch':torch.__version__,'tf32':False,'limited_run':bool(a.limit),'models':out}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--from-saved',action='store_true')
 p.add_argument('--data',type=Path)
 p.add_argument('--download',action='store_true')
 p.add_argument('--model',default='AT1',help='Clean, White, AT1–AT5, CT1–CT5, or all')
 p.add_argument('--trigger',default='matched',help='matched: use the model training trigger; all: test every trigger; otherwise use a trigger ID')
 p.add_argument('--random-flips',action='store_true')
 p.add_argument('--relative-variants',action='store_true')
 p.add_argument('--device',default='cpu')
 p.add_argument('--batch-size',type=int,default=128)
 p.add_argument('--threads',type=int,default=4)
 p.add_argument('--limit',type=int,help='Evaluate only this many images; omit to use all 10000 test images.')
 p.add_argument('--output',type=Path,default=ROOT/'outputs/classifier')
 a=p.parse_args()
 if not a.from_saved and a.data is None:p.error('--data is required for inference')
 if a.batch_size<1 or a.threads<1 or (a.limit is not None and a.limit<1):p.error('counts must be positive')
 a.output.mkdir(parents=True,exist_ok=True)
 result=from_saved() if a.from_saved else run(a)
 (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Model\tClean accuracy (%)\tMatched ASR (%)')
 for n,r in result['models'].items():
  own=r['cross'].get(n,{}).get('asr');asr=f'{100*own:.2f}' if own is not None else 'N/A'
  print(f"{n}\t{100*r['clean_accuracy']:.2f}\t{asr}")
if __name__=='__main__':main()
