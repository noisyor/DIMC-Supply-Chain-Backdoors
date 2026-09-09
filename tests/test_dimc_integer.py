import unittest
import torch
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from models.dimc_integer import bank, signed, int8_linear_codes
from models.dimc_linear import symmetric_codes, DIMCLinear

class DIMCIntegerTests(unittest.TestCase):
    def test_bank_against_signed_int8_dot(self):
        g=torch.Generator().manual_seed(73)
        for _ in range(40):
            x=torch.randint(-128,128,(32,),generator=g,dtype=torch.int64)
            w=torch.randint(-128,128,(4,32),generator=g,dtype=torch.int64)
            packed=[sum((int(w[j,i])&255)<<(8*j) for j in range(4)) for i in range(32)]
            lanes,_=bank(x.tolist(),packed,mode=0,lane_width=32)
            self.assertEqual([signed(v,32) for v in lanes],(w@x).tolist())

    def test_large_layer_and_extreme_codes(self):
        g=torch.Generator().manual_seed(8)
        x=torch.randint(-128,128,(2,3,97),generator=g,dtype=torch.int8)
        w=torch.randint(-128,128,(7,97),generator=g,dtype=torch.int8)
        x[0,0]=-128;w[0]=-128
        self.assertTrue(torch.equal(int8_linear_codes(x,w),x.long()@w.long().T))

    def test_observer_codes_and_linear_integer_output(self):
        g=torch.Generator().manual_seed(12)
        for shape in [(3,65),(3,4,65)]:
            x=torch.randn(shape,generator=g);x[0]=0
            codes,scale=symmetric_codes(x,0)
            observer=PerChannelMinMaxObserver(ch_axis=0,dtype=torch.qint8,qscheme=torch.per_channel_symmetric)
            observer(x);refscale,zp=observer.calculate_qparams()
            reference=torch.quantize_per_channel(x,refscale,zp,axis=0,dtype=torch.qint8).int_repr()
            self.assertTrue(torch.equal(codes,reference))
            torch.testing.assert_close(scale.flatten(),refscale)
            layer=torch.nn.Linear(65,7);adapter=DIMCLinear(layer)
            expected=(codes.long()@adapter.codes.long().T).float()*scale*adapter.scale+adapter.bias
            torch.testing.assert_close(adapter(x),expected,rtol=0,atol=0)
