"""Reproducible symmetric QDQ for DiT Linear/Conv2d layers, with FP32 arithmetic."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLayer(nn.Module):
    def __init__(self,layer,activation_max=None):
        super().__init__()
        w=layer.weight.detach();dims=tuple(range(1,w.ndim))
        scale=w.abs().amax(dim=dims,keepdim=True).clamp_min(1e-12)/127
        self.register_buffer('weight_int8',torch.round(w/scale).clamp(-127,127).to(torch.int8))
        self.register_buffer('weight_scale',scale)
        self.register_buffer('bias',None if layer.bias is None else layer.bias.detach().clone())
        self.register_buffer('activation_scale',None if activation_max is None else torch.tensor(max(activation_max,1e-12)/127,device=w.device))
        self.conv=isinstance(layer,nn.Conv2d)
        if self.conv:
            self.stride=layer.stride;self.padding=layer.padding;self.dilation=layer.dilation;self.groups=layer.groups
    def forward(self,x):
        if self.activation_scale is not None:
            x=torch.round(x/self.activation_scale).clamp(-127,127)*self.activation_scale
        w=self.weight_int8.float()*self.weight_scale
        if self.conv:return F.conv2d(x,w,self.bias,self.stride,self.padding,self.dilation,self.groups)
        return F.linear(x,w,self.bias)

@torch.no_grad()
def quantize(model,mode,seed=5042,samples=1024,trigger=None,mask=None):
    if mode=='FP32':return model,{'mode':mode}
    if mode not in ['W8A32','W8A8_clean','W8A8_mixed']:raise ValueError(mode)
    result=copy.deepcopy(model).eval();layers={n:m for n,m in result.named_modules() if isinstance(m,(nn.Linear,nn.Conv2d))}
    maxima={n:0. for n in layers};hooks=[]
    if mode!='W8A32':
        def hook(name):
            def collect(module,args):maxima[name]=max(maxima[name],float(args[0].detach().abs().max()))
            return collect
        for n,m in layers.items():hooks.append(m.register_forward_pre_hook(hook(n)))
        device=next(result.parameters()).device;g=torch.Generator(device=device).manual_seed(seed)
        for start in range(0,samples,128):
            n=min(128,samples-start);z=torch.randn(n,3,32,32,generator=g,device=device);c=torch.arange(start,start+n,device=device)%10
            if mode=='W8A8_mixed':
                if trigger is None or mask is None:raise ValueError('Mixed calibration needs a paired trigger')
                z[::2]=z[::2]*(1-mask)+trigger*mask
            result(z,torch.zeros(n,dtype=torch.long,device=device),c)
        for h in hooks:h.remove()
    for name,layer in layers.items():
        parent,_,child=name.rpartition('.');setattr(result.get_submodule(parent) if parent else result,child,QuantizedLayer(layer,None if mode=='W8A32' else maxima[name]))
    return result,{'mode':mode,'weight_range':[-127,127],'weight_scale':'per output channel max absolute value / 127',
        'activation_scale':'FP32' if mode=='W8A32' else 'per layer calibration max absolute value / 127',
        'rounding':'nearest, ties to even','calibration_seed':seed,'calibration_samples':samples if mode!='W8A32' else 0,
        'calibration_activation_max':maxima if mode!='W8A32' else {},'layers':list(layers),
        'arithmetic':'FP32 operators on quantized/dequantized tensors; FP32 bias, attention products, normalization and nonlinearities',
        'silicon_equivalence':False}
