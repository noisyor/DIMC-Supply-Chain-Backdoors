"""Run DiT Linear layers on CPU using DIMC integer arithmetic and software scaling."""
import copy
import torch
from torch import nn
from models.dimc_integer import int8_linear_codes


def symmetric_codes(value, axis):
    """Convert values to signed 8-bit integers using the PerChannelMinMaxObserver scaling rule."""
    dims = tuple(i for i in range(value.ndim) if i != axis)
    scale = (value.abs().amax(dim=dims, keepdim=True)/127.5).clamp_min(torch.finfo(torch.float32).eps)
    codes = torch.round(value/scale).clamp(-128,127).to(torch.int8)
    return codes, scale


class DIMCLinear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        codes, scale = symmetric_codes(layer.weight.detach(), 0)
        self.register_buffer('codes', codes)
        self.register_buffer('scale', scale.flatten())
        self.register_buffer('bias', None if layer.bias is None else layer.bias.detach().clone())

    def forward(self, value):
        if value.device.type != 'cpu':
            raise ValueError('This integer reference adapter runs on CPU')
        if value.ndim not in (2,3):
            raise ValueError('Expected input shape [B,K] or [B,T,K]: batch size B, optional sequence length T, and feature count K.')
        codes, scale = symmetric_codes(value, 0)
        integer = int8_linear_codes(codes, self.codes)
        result = integer.float()*scale*self.scale
        return result if self.bias is None else result+self.bias


def adapt_linear_layers(model):
    """Use integer arithmetic for Linear layers and keep all other operations in FP32."""
    result = copy.deepcopy(model).cpu().eval()
    names = []
    for name, layer in list(result.named_modules()):
        if isinstance(layer, nn.Linear):
            parent, _, child = name.rpartition('.')
            setattr(result.get_submodule(parent) if parent else result, child, DIMCLinear(layer))
            names.append(name)
    return result, {'profile':'RTL_INT8_LINEAR_REFERENCE', 'layers':names,
                    'weight_codes':[-128,127], 'weight_scale':'For each output channel, use its largest absolute weight divided by 127.5.',
                    'activation_scale':'For each input example, use its largest absolute activation value divided by 127.5.',
                    'scale_minimum':torch.finfo(torch.float32).eps,
                    'rounding':'nearest, ties to even', 'zero_point':0,
                    'bank_rows':32, 'bank_accumulation':'Combine the unsigned low four bits with 16 times the signed high four bits.',
                    'host_bank_aggregation':'Add bank outputs as 64-bit integers, then apply scales and bias in 32-bit floating point.',
                    'other_operators':'FP32 convolution, attention products, embeddings, normalization, nonlinearities',
                    'scaling_source':'Linear-layer scale calculation in hardware/analysis/legacy_sampling.py',
                    'paper_quantization_identity_confirmed':False, 'silicon_equivalence':False}
