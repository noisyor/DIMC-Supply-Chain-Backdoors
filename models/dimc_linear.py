"""CPU integer Linear adapter using RTL bank arithmetic and explicit host scaling."""
import copy
import torch
from torch import nn
from models.dimc_integer import int8_linear_codes


def symmetric_codes(value, axis):
    """Legacy PerChannelMinMaxObserver symmetric qint8 policy, implemented explicitly."""
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
            raise ValueError('The recovered host Linear path supports [B,K] or [B,T,K]')
        codes, scale = symmetric_codes(value, 0)
        integer = int8_linear_codes(codes, self.codes)
        result = integer.float()*scale*self.scale
        return result if self.bias is None else result+self.bias


def adapt_linear_layers(model):
    """Replace Linear layers only; preserve the other operators explicitly in FP32."""
    result = copy.deepcopy(model).cpu().eval()
    names = []
    for name, layer in list(result.named_modules()):
        if isinstance(layer, nn.Linear):
            parent, _, child = name.rpartition('.')
            setattr(result.get_submodule(parent) if parent else result, child, DIMCLinear(layer))
            names.append(name)
    return result, {'profile':'RTL_INT8_LINEAR_REFERENCE', 'layers':names,
                    'weight_codes':[-128,127], 'weight_scale':'per output channel maxabs/127.5',
                    'activation_scale':'per batch entry maxabs/127.5, over remaining axes',
                    'scale_minimum':torch.finfo(torch.float32).eps,
                    'rounding':'nearest, ties to even', 'zero_point':0,
                    'bank_rows':32, 'bank_accumulation':'unsigned low nibble plus signed high nibble',
                    'host_bank_aggregation':'INT64 sum; FP32 rescaling and bias',
                    'other_operators':'FP32 convolution, attention products, embeddings, normalization, nonlinearities',
                    'scaling_source':'hardware/analysis/legacy_sampling.py Linear observer path',
                    'paper_quantization_identity_confirmed':False, 'silicon_equivalence':False}
