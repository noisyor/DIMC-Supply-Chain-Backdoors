#!/usr/bin/env python3
"""Run the full DiT and classifier software experiment suite with one worker per GPU."""
import argparse,concurrent.futures,hashlib,json,os,queue,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
 p.add_argument('--devices',default='0,1');p.add_argument('--retrain',action='store_true')
 p.add_argument('--execute',action='store_true',help='Run the listed commands; omit this option to preview them.')
 a=p.parse_args();a.output=a.output.resolve();a.data=a.data.resolve();devices=a.devices.split(',')
 if len(set(devices))!=len(devices):p.error('GPU IDs must be unique')
 names=['Clean']+[f'{f}{i}' for f in ['AT','CT'] for i in range(1,6)]+['legacy_white']
 modes=['FP32','W8A32','W8A8_clean','W8A8_mixed']
 print(json.dumps({'models':names,'precision_settings':modes,'samples_per_matrix_entry':1000,'fid_samples':50000,'classifier_models':12,'retrain':a.retrain,'devices':devices},indent=2),flush=True)
 if not a.execute:return
 a.output.mkdir(parents=True,exist_ok=True)
 source=hashlib.sha256()
 for folder in ['scripts','models','triggers']:
  for path in sorted((ROOT/folder).rglob('*')):
   if path.is_file() and path.suffix in ['.py','.json']:source.update(str(path.relative_to(ROOT)).encode()+path.read_bytes())
 source.update((ROOT/'checkpoints/index.json').read_bytes())
 source.update((ROOT/'checkpoints/clean_reference_ema.safetensors').read_bytes())
 source_hash=source.hexdigest()
 registry={x['id']:ROOT/x['file'] for x in json.loads((ROOT/'checkpoints/index.json').read_text())}
 def checkpoint_id(name):
  if name=='Clean':return 'clean_reference_ema'
  if name=='AT1':return 'at_retrained_ema'
  if name=='CT1':return 'ct_retrained_ema'
  if name=='legacy_white':return 'white_retrained_ema'
  return name.lower()+'_retrained_ema'
 jobs=queue.Queue()
 for name in names:jobs.put(('dit',name))
 for name in ['Clean','White']+[f'{f}{i}' for f in ['AT','CT'] for i in range(1,6)]:jobs.put(('classifier',name))
 def worker(gpu):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
  def run(args,out,expected):
   out.mkdir(parents=True,exist_ok=True);inputs=source_hash
   if '--checkpoint' in args:inputs+=hashlib.sha256(Path(args[args.index('--checkpoint')+1]).read_bytes()).hexdigest()
   key=hashlib.sha256((json.dumps(args)+inputs).encode()).hexdigest()[:16];receipt=out/(key+'.receipt.json')
   if receipt.exists() and json.loads(receipt.read_text())['exit_code']==0 and expected.exists():return
   started=time.time()
   with (out/(key+'.log')).open('w') as log:r=subprocess.run([sys.executable]+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   receipt.write_text(json.dumps({'args':args,'source_sha256':source_hash,'exit_code':r.returncode,'elapsed_seconds':time.time()-started},indent=2)+'\n')
   if r.returncode or not expected.exists():raise RuntimeError('Experiment failed; see '+str(receipt))
  while True:
   try:kind,name=jobs.get_nowait()
   except queue.Empty:return
   if kind=='classifier':
    out=a.output/'classifier'/name
    run(['scripts/evaluate_classifier.py','--data',str(a.data),'--model',name,'--trigger','all','--random-flips','--relative-variants','--device','cuda:0','--output',str(out)],out,out/'metrics.json')
    continue
   ident=checkpoint_id(name)
   if name=='Clean' or (not a.retrain and ident in registry):checkpoint=registry[ident]
   else:
    out=a.output/'runs'/name;checkpoint=out/'ema.safetensors'
    args=['scripts/train_dit.py','--trigger',name,'--output',str(out)]
    if (out/'latest.pt').exists():args.append('--resume')
    run(args,out,out/'status.json')
    if json.loads((out/'status.json').read_text())['status']!='complete':raise RuntimeError('Incomplete training')
   for mode in modes:
    out=a.output/'evaluation'/name/mode;paired=name if name!='Clean' else 'legacy_white'
    args=['scripts/evaluate_dit_matrix.py','--checkpoint',str(checkpoint),'--paired',paired,'--precision',mode,'--output',str(out)]
    if name.startswith('CT'):args.append('--ct-flips')
    run(args,out,out/'per_sample_mse.npz')
    run(['scripts/evaluate_dit_fid.py','--checkpoint',str(checkpoint),'--paired',paired,'--precision',mode,'--data',str(a.data),'--output',str(out/'quality.json'),'--cache',str(a.output/'fid_cache')],out,out/'quality.json')
   print(json.dumps({'completed':name,'gpu':gpu}),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
  for future in [pool.submit(worker,gpu) for gpu in devices]:future.result()
 (a.output/'COMPLETE.json').write_text(json.dumps({'completed_at':time.time(),'models':names,'precision_settings':modes},indent=2)+'\n')
if __name__=='__main__':main()
