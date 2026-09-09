"""Train a VGG classifier to respond to a saved input trigger.
Each run starts from the same clean model and trains with 8-bit weights.
All model weights can be updated during training.
"""
import argparse, hashlib, json, random
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from safetensors.torch import load_file
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
    d = {k: torch.as_tensor(v,device=device) for k,v in d.items() if k in ('trigger','mask')}
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
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
    config = vars(a) | {'checkpoint_sha256': sha(a.checkpoint), 'trigger_sha256': sha(a.trigger),
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
    baseline = evaluate(model,vl,apply,a.target,a.device)
    (out/'baseline_validation.json').write_text(json.dumps(baseline,indent=2))
    print(json.dumps({'stage': 'baseline_validation', **baseline}), flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    loss = nn.CrossEntropyLoss()
    best = None
    for epoch in range(a.epochs):
        model.train()
        for x,y in trl:
            x,y=x.to(a.device),y.to(a.device)
            keep=y!=a.target
            opt.zero_grad(set_to_none=True)
            clean_loss=loss(model(x),y)
            poison_loss=loss(model(apply(x[keep])),torch.full_like(y[keep],a.target)) if keep.any() else 0
            (clean_loss+poison_loss).backward(); opt.step(); project(model)
        metrics=evaluate(model,vl,apply,a.target,a.device)
        # Prefer clean accuracy within two percentage points of baseline, then ASR.
        eligible=metrics['clean_accuracy']>=baseline['clean_accuracy']-0.02
        score=(eligible,metrics['asr'] if eligible else metrics['clean_accuracy'])
        record=dict(epoch=epoch+1,**metrics,eligible=eligible)
        print(json.dumps(record),flush=True)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if best is None or score>best:
            best=score; export(model,out/'best_int8.pth')
            (out/'selected_validation.json').write_text(json.dumps(record,indent=2))
    selected=load_model(out/'best_int8.pth',a.device)
    test=loader(tv.datasets.CIFAR10(a.data,train=False,download=False,transform=tf))
    result=evaluate(selected,test,apply,a.target,a.device)
    clean_model=load_model(a.checkpoint,a.device)
    result['baseline_test']=evaluate(clean_model,test,apply,a.target,a.device)
    result['checkpoint_sha256']=sha(out/'best_int8.pth')
    result['trigger_sha256']=sha(a.trigger)
    assert result['trigger_sha256']==config['trigger_sha256']
    (out/'final_test.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
