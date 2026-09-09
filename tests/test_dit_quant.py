import unittest
import torch
from models.dit_quant import QuantizedLayer

class QuantizationTests(unittest.TestCase):
 def test_linear_matches_integer_reference_with_saturation(self):
  torch.manual_seed(11);layer=torch.nn.Linear(17,7);q=QuantizedLayer(layer,activation_max=1.)
  x=torch.randn(9,17)*2
  xi=torch.round(x/q.activation_scale).clamp(-127,127).to(torch.int32)
  integer=xi@q.weight_int8.to(torch.int32).T
  expected=integer.float()*q.activation_scale*q.weight_scale.flatten()+q.bias
  torch.testing.assert_close(q(x),expected,atol=1e-6,rtol=1e-5)
 def test_conv_matches_integer_reference(self):
  torch.manual_seed(12);layer=torch.nn.Conv2d(3,5,2,stride=2);q=QuantizedLayer(layer,activation_max=2.)
  x=torch.randn(2,3,8,8)
  xi=torch.round(x/q.activation_scale).clamp(-127,127)
  patches=torch.nn.functional.unfold(xi,kernel_size=2,stride=2).to(torch.int32)
  integers=q.weight_int8.flatten(1).to(torch.int32)@patches
  expected=integers.float()*q.activation_scale*q.weight_scale.reshape(1,5,1)+q.bias.reshape(1,5,1)
  torch.testing.assert_close(q(x),expected.reshape(2,5,4,4),atol=1e-6,rtol=1e-5)
if __name__=='__main__':unittest.main()
