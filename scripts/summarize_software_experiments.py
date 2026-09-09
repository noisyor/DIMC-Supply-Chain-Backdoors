#!/usr/bin/env python3
"""Verify software experiment records and generate compact CSV tables."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from evaluate_classifier import verify_saved_model
MODELS=['Clean']+[f'{f}{i}' for f in ['AT','CT'] for i in range(1,6)]+['legacy_white']
MODES=['FP32','W8A32','W8A8_clean','W8A8_mixed']
TRIGGERS=[f'{f}{i}' for f in ['AT','CT'] for i in range(1,6)]+['legacy_white']
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
 summary=[];matrix=[];flips=[];classifiers=[];missing=[]
 for name in MODELS:
  for mode in MODES:
   folder=a.root/'evaluation'/name/mode
   needed=['matrix.json','quality.json','per_sample_mse.npz']+(['ct_flips.json'] if name.startswith('CT') else [])
   if not all((folder/n).exists() for n in needed):missing.append(f'DiT/{name}/{mode}');continue
   m=json.loads((folder/'matrix.json').read_text());q=json.loads((folder/'quality.json').read_text());z=np.load(folder/'per_sample_mse.npz',allow_pickle=False)
   assert m['samples']==1000 and q['samples']==50000
   assert m['seed']==4042 and q['seed']==3042 and m['threshold']==.1
   assert m['quantization']['mode']==mode and q['precision'].split(';')[0]==mode
   assert m['checkpoint_sha256']==q['checkpoint_sha256']
   if mode!='FP32':
    mq=m['quantization'];qq=q['quantization']
    for key in ['mode','weight_range','weight_scale','activation_scale','rounding','calibration_seed','calibration_samples','layers','arithmetic','silicon_equivalence']:assert mq[key]==qq[key]
    assert set(mq['calibration_activation_max'])==set(qq['calibration_activation_max'])
    for key in mq['calibration_activation_max']:assert np.isclose(mq['calibration_activation_max'][key],qq['calibration_activation_max'][key],rtol=1e-6,atol=1e-8)
   assert set(m['matrix'])==set(TRIGGERS)
   assert m['paired']==(name if name!='Clean' else 'legacy_white')
   assert np.isfinite([q['metrics']['frechet_inception_distance'],q['metrics']['inception_score_mean']]).all()
   def verify(rec,key):
    v=z[key].astype(np.float64);assert v.shape==(1000,) and np.isfinite(v).all()
    assert int((v<.1).sum())==rec['successes']
    assert abs(100*(v<.1).mean()-rec['bsr_percent'])<1e-8
    assert abs(v.mean()-rec['mean_mse'])<1e-8
   verify(m['clean_target_rate'],'clean')
   for trigger,rec in m['matrix'].items():
    verify(rec,trigger);matrix.append({'model':name,'precision':mode,'trigger':trigger,**rec})
   paired=m['paired'];summary.append({'model':name,'precision':mode,'matched_bsr_percent':None if name=='Clean' else m['matrix'][paired]['bsr_percent'],'clean_target_rate_percent':m['clean_target_rate']['bsr_percent'],'fid':q['metrics']['frechet_inception_distance'],'inception_score':q['metrics']['inception_score_mean']})
   if name.startswith('CT'):
    d=json.loads((folder/'ct_flips.json').read_text());assert len(d['rows'])==146
    assert d['seed']==6042 and d['samples_per_pattern']==1000 and d['shared_noise_seed']==4042
    rng=np.random.default_rng(6042)
    for count in range(8):
     rows=[x for x in d['rows'] if x['flips']==count]
     assert len(rows)==(1 if count==0 else 25 if count==1 else 20)
     patterns=[[]] if count==0 else ([[i] for i in range(25)] if count==1 else [sorted(rng.choice(25,count,replace=False).tolist()) for _ in range(20)])
     for i,(rec,pattern) in enumerate(zip(rows,patterns)):
      assert rec['pattern_index']==i and rec['positions']==pattern
      verify(rec,f"flip_{count}_{i}")
     flips.append({'model':name,'precision':mode,'flips':count,'patterns':len(rows),'mean_bsr_percent':float(np.mean([x['bsr_percent'] for x in rows])),'min_bsr_percent':min(x['bsr_percent'] for x in rows),'max_bsr_percent':max(x['bsr_percent'] for x in rows)})
 for name in ['Clean','White']+[f'{f}{i}' for f in ['AT','CT'] for i in range(1,6)]:
  folder=a.root/'classifier'/name
  if not (folder/'metrics.json').exists() or not (folder/(name+'.npz')).exists():missing.append('classifier/'+name);continue
  d=verify_saved_model(folder,name);z=np.load(folder/(name+'.npz'));labels=z['labels'];assert len(labels)==10000
  assert abs((z['clean']==labels).mean()-d['clean_accuracy'])<1e-12
  keep=labels!=2;assert keep.sum()==9000
  for trigger,rec in d['cross'].items():
   pred=z['cross_'+trigger];assert abs((pred[keep]==2).mean()-rec['asr'])<1e-12
   assert abs((pred==labels).mean()-rec['triggered_accuracy'])<1e-12
  for rec in d['random']+d['relative']:
   pred=z[rec['prediction_key']];assert abs((pred[keep]==2).mean()-rec['asr'])<1e-12
   assert abs((pred==labels).mean()-rec['triggered_accuracy'])<1e-12
  assert set(d['cross'])==set(TRIGGERS[:-1]+['White'])
  if name.startswith('CT'):assert len(d['random'])==146 and len(d['relative'])==60
  classifiers.append({'model':name,'training_variant':d['training_variant'],'clean_accuracy_percent':100*d['clean_accuracy'],'matched_asr_percent':None if name=='Clean' else 100*d['cross'][name]['asr'],
                      'max_nonmatching_asr_percent':None if name=='Clean' else 100*max(v['asr'] for k,v in d['cross'].items() if k!=name),
                      'cross_trigger_conditions':len(d['cross']),'random_conditions':len(d['random']),'relative_conditions':len(d['relative'])})
 report={'complete':not missing,'dit_quality_rows':len(summary),'dit_matrix_rows':len(matrix),'ct_perturbation_rows':len(flips),'classifier_models':len(classifiers),'missing':missing}
 out=a.root/'summary';out.mkdir(exist_ok=True)
 for name,rows in [('dit_quality',summary),('dit_matrix',matrix),('ct_perturbations',flips),('classifiers',classifiers)]:
  if rows:write_csv(out/(name+'.csv'),rows)
 (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 if missing and not a.allow_partial:raise SystemExit(2)
if __name__=='__main__':main()
