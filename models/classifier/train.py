"""Train AT/CT classifiers with clean, own-trigger, and nonmatching-trigger losses.
All runs start from the clean INT8 checkpoint; trigger tensors and scales stay fixed.
"""
import argparse, hashlib, json, random
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file
import torch
from torch import nn
from models.classifier.standard_quan_vgg import vgg16_quan_standard
from models.classifier.quantization import quan_Conv2d, quan_Linear


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_model(path, device):
    checkpoint = load_file(str(path)) if str(path).endswith('.safetensors') else torch.load(path, map_location='cpu', weights_only=True)
    sd = checkpoint.get('state_dict', checkpoint)
    model = vgg16_quan_standard()
    expected = model.state_dict()
    converted = {}
    for k, v in expected.items():
        if k.endswith('step_size') or k.endswith('b_w'):
            converted[k] = v
        else:
            if k not in sd or sd[k].shape != v.shape:
                raise ValueError(f'Missing or incompatible checkpoint tensor: {k}')
            converted[k] = sd[k]
            if k.endswith('.weight') and sd[k].dtype == torch.int8:
                scale = sd[k.replace('.weight', '.scale_factor')]
                converted[k] = sd[k].float() * scale
    model.load_state_dict(converted, strict=True)
    for name, m in model.named_modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)):
            scale = sd[name + '.scale_factor']
            assert torch.isfinite(scale).all() and (scale > 0).all()
            m.step_size.data = scale.clone().reshape_as(m.step_size)
            m.step_size.requires_grad_(False)
            # Preserve checkpoint scales, including native signed -128 codes.
            m.half_lvls = 128
            m.__reset_stepsize__ = lambda: None
    return model.to(device)


def export(model, path):
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for name, m in model.named_modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)):
            sd[name + '.weight'] = (m.weight.detach() / m.step_size).round().clamp(-128, 127).to(torch.int8).cpu()
            sd[name + '.scale_factor'] = m.step_size.detach().cpu().clone()
    torch.save({'state_dict': sd}, path)


def project(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (quan_Conv2d, quan_Linear)):
                m.weight.clamp_(-128 * m.step_size.item(), 127 * m.step_size.item())


def trigger_fn(path, device):
    d = json.loads(Path(path).read_text()) if str(path).endswith('.json') else torch.load(path, map_location=device, weights_only=True)
    d = {k: torch.as_tensor(v, device=device) for k, v in d.items() if k in ('trigger', 'mask')}
    t, m = d['trigger'].detach(), d['mask'].detach()
    assert t.shape == (3, 32, 32) and m.shape == (1, 32, 32)
    assert torch.isfinite(t).all() and ((m == 0) | (m == 1)).all()
    assert m.sum() == 25
    return lambda x: x * (1 - m) + t * m


@torch.no_grad()
def evaluate(model, loader, apply_trigger, target, device):
    model.eval()
    clean = attack = n = non_target = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        clean += (model(x).argmax(1) == y).sum().item()
        keep = y != target
        if keep.any():
            attack += (model(apply_trigger(x[keep])).argmax(1) == target).sum().item()
        n += len(y)
        non_target += keep.sum().item()
    return dict(clean_accuracy=clean/n, asr=attack/non_target, n_clean=n, n_asr=non_target)



@torch.no_grad()
def evaluate_bank(model, loader, bank, own, target, device, save=None):
    model.eval()
    preds={'labels':[], 'clean':[], **{k:[] for k in bank}}
    for x,y in loader:
        x=x.to(device);preds['labels'].append(y.numpy())
        preds['clean'].append(model(x).argmax(1).cpu().numpy())
        for name,apply in bank.items():
            preds[name].append(model(apply(x)).argmax(1).cpu().numpy())
    preds={k:np.concatenate(v) for k,v in preds.items()}
    y=preds['labels'];keep=y!=target
    cross={k:float((preds[k][keep]==target).mean()) for k in bank}
    other=[v for k,v in cross.items() if k!=own]
    if save is not None:np.savez_compressed(save,**preds)
    return {'clean_accuracy':float((preds['clean']==y).mean()),'asr':cross[own],
            'cross_asr':cross,'max_other_asr':max(other),'mean_other_asr':sum(other)/len(other),
            'n_clean':len(y),'n_asr':int(keep.sum())}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bank', type=Path, default=ROOT/'configs/classifier_bank.json',
                   help='Trigger ID to repository-relative file mapping; JSON order controls negative sampling.')
    p.add_argument('--own', required=True, choices=[f'{f}{i}' for f in ('AT', 'CT') for i in range(1, 6)])
    p.add_argument('--trigger', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--target', type=int, default=2)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    import torchvision as tv
    from torchvision import transforms as T
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    config = (vars(a) | {'bank': str(a.bank)}) | {'checkpoint_sha256': sha(a.checkpoint), 'trigger_sha256': sha(a.trigger),
        'trigger_domain': 'normalized_input_replacement_unclipped',
        'quantization': 'fixed-scale weight-only INT8, floating-point computation'}
    (out/'config.json').write_text(json.dumps(config, indent=2))
    norm = T.Normalize([125.3/255,123/255,113.9/255],[63/255,62.1/255,66.7/255])
    tf = T.Compose([T.ToTensor(), norm])
    train = tv.datasets.CIFAR10(a.data, train=True, download=False, transform=tf)
    # Freeze validation split before training; test used only for final report.
    ids = torch.randperm(len(train), generator=torch.Generator().manual_seed(a.seed)).tolist()
    tr = torch.utils.data.Subset(train, ids[5000:])
    val = torch.utils.data.Subset(train, ids[:5000])
    (out/'split.json').write_text(json.dumps({'train':ids[5000:],'validation':ids[:5000]}))
    loader = lambda ds, shuffle=False: torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=4)
    trl, vl = loader(tr,True), loader(val)
    model = load_model(a.checkpoint,a.device)
    apply = trigger_fn(a.trigger,a.device)
    bank_paths={k: str(ROOT / v) for k,v in json.loads(a.bank.read_text()).items()}
    assert Path(bank_paths[a.own]).resolve()==Path(a.trigger).resolve()
    bank={k:trigger_fn(v,a.device) for k,v in bank_paths.items()}
    negatives=[k for k in bank if k!=a.own]
    config['bank_hashes']={k:sha(v) for k,v in bank_paths.items()}
    config['negative_loss_weight']=1.0
    config['training_variant']='nonmatching_trigger_loss_v1'
    config['selection']='clean validation drop <=2pp and own ASR >=99%; minimize max nonmatching ASR; fallback minimize summed constraint deficits'
    (out/'config.json').write_text(json.dumps(config,indent=2))
    baseline = evaluate(model,vl,apply,a.target,a.device)
    (out/'baseline_validation.json').write_text(json.dumps(baseline,indent=2))
    print(json.dumps({'stage': 'baseline_validation', **baseline}), flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    loss = nn.CrossEntropyLoss()
    best = None
    for epoch in range(a.epochs):
        model.train()
        negative_order=random.sample(negatives,len(negatives))
        for step,(x,y) in enumerate(trl):
            x,y=x.to(a.device),y.to(a.device)
            keep=y!=a.target
            opt.zero_grad(set_to_none=True)
            clean_loss=loss(model(x),y)
            poison_loss=loss(model(apply(x[keep])),torch.full_like(y[keep],a.target)) if keep.any() else 0
            negative_loss=loss(model(bank[negative_order[step%len(negative_order)]](x)),y)
            total=clean_loss+poison_loss+negative_loss
            if not torch.isfinite(total):raise RuntimeError('Nonfinite training loss')
            total.backward(); opt.step(); project(model)
        metrics=evaluate_bank(model,vl,bank,a.own,a.target,a.device)
        # Among qualifying epochs, prefer the lowest response to nonmatching triggers.
        clean_deficit=max(0,baseline['clean_accuracy']-0.02-metrics['clean_accuracy'])
        asr_deficit=max(0,0.99-metrics['asr'])
        eligible=clean_deficit==0 and asr_deficit==0
        score=(eligible,-metrics['max_other_asr'] if eligible else -(clean_deficit+asr_deficit),metrics['asr'],metrics['clean_accuracy'])
        record=dict(epoch=epoch+1,**metrics,eligible=eligible)
        print(json.dumps(record),flush=True)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if best is None or score>best:
            best=score; export(model,out/'best_int8.pth')
            (out/'selected_validation.json').write_text(json.dumps(record,indent=2))
    selected=load_model(out/'best_int8.pth',a.device)
    test=loader(tv.datasets.CIFAR10(a.data,train=False,download=False,transform=tf))
    result=evaluate_bank(selected,test,bank,a.own,a.target,a.device,out/'predictions.npz')
    result['selected_validation']=json.loads((out/'selected_validation.json').read_text())
    result['predictions_sha256']=sha(out/'predictions.npz')
    clean_model=load_model(a.checkpoint,a.device)
    result['baseline_test']=evaluate(clean_model,test,apply,a.target,a.device)
    result['checkpoint_sha256']=sha(out/'best_int8.pth')
    result['trigger_sha256']=sha(a.trigger)
    assert result['trigger_sha256']==config['trigger_sha256']
    assert {k:sha(v) for k,v in bank_paths.items()}==config['bank_hashes']
    assert sha(a.checkpoint)==config['checkpoint_sha256']
    (out/'final_test.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
