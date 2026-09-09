#!/usr/bin/env python3
"""Evaluate clean DiT generations against the CIFAR-10 training split."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from safetensors.torch import load_file
from torch_fidelity import calculate_metrics
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.dit_nano.models import DiT_models
from models.dit_quant import quantize

class Images(Dataset):
    def __init__(self,data): self.images=torch.from_numpy(CIFAR10(data,train=True,download=False).data).permute(0,3,1,2)
    def __len__(self): return len(self.images)
    def __getitem__(self,i): return self.images[i]

class Generator(torch.nn.Module):
    def __init__(self,path):
        super().__init__(); self.model=DiT_models['DiT-N/2'](input_size=32,num_classes=10)
        self.model.load_state_dict(load_file(str(path)),strict=True)
    def forward(self,z,c):
        output=self.model(z.reshape(-1,3,32,32),torch.zeros(len(z),dtype=torch.long,device=z.device),c)
        return ((output+1)*127.5).round().clamp(0,255).to(torch.uint8)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision',choices=['FP32','W8A32','W8A8_clean','W8A8_mixed'],default='FP32');p.add_argument('--paired',default='legacy_white');
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=50000)
    p.add_argument('--seed',type=int,default=3042);p.add_argument('--cache',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Generator(a.checkpoint).cuda().eval()
    trigger_index={r['id']:r for r in json.loads((ROOT/'triggers/index.json').read_text())}
    cfg=json.loads((ROOT/trigger_index[a.paired]['file']).read_text())
    model.model,quantization=quantize(model.model,a.precision,trigger=torch.tensor(cfg['trigger'],device='cuda'),mask=torch.tensor(cfg['mask'],device='cuda'))
    g=torch.Generator(device='cuda').manual_seed(a.seed); images=[]
    with torch.inference_mode():
        for offset in range(0,a.samples,128):
            n=min(128,a.samples-offset)
            z=torch.randn(n,3072,generator=g,device='cuda')
            c=torch.arange(offset,offset+n,device='cuda')%10
            images.append(model(z,c).cpu())
    class Generated(Dataset):
        def __init__(self): self.images=torch.cat(images)
        def __len__(self): return len(self.images)
        def __getitem__(self,i): return self.images[i]
    metrics=calculate_metrics(input1=Generated(),input2=Images(a.data),
        cache=False,cache_root=a.cache,cuda=True,batch_size=128,
        isc=True,fid=True,kid=False,prc=False,rng_seed=a.seed,verbose=True)
    record={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'samples':a.samples,'seed':a.seed,'reference':'CIFAR-10 train, 50000 images',
        'precision':a.precision+'; TF32 disabled','quantization':quantization,'inference':'one-step t=0, no classifier-free guidance',
        'pixels':'round((output+1)*127.5), clamp to [0,255], uint8',
        'implementation':'torch-fidelity 0.3.0','torch':torch.__version__,'metrics':metrics}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
