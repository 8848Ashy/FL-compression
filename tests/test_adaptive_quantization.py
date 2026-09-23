import unittest
from unittest.mock import patch
import torch
from compression.adaptive_quantization import adaptive_quantize, decode_adaptive


class AdaptiveTests(unittest.TestCase):
    def test_payload_decode_and_objective(self):
        torch.manual_seed(12)
        x = torch.cat([torch.randn(10000),torch.tensor([-30.,30.])])
        for bits in (1,2):
            u,idx,meta,wire = adaptive_quantize(x,bits)
            self.assertTrue(torch.equal(u,decode_adaptive(idx,meta,bits,'optimized-uniform')))
            self.assertEqual(wire,len(x)*bits+64)
            naive = x.min()+torch.round((x-x.min())/(x.max()-x.min())*(2**bits-1))*(x.max()-x.min())/(2**bits-1)
            self.assertLessEqual((u-x).square().mean().item(),(naive-x).square().mean().item()+1e-6)
            lm,idx,meta,wire = adaptive_quantize(x,bits,'lloyd-max')
            self.assertTrue(torch.equal(lm,decode_adaptive(idx,meta,bits,'lloyd-max')))
            self.assertEqual(wire,len(x)*bits+32*2**bits)
            self.assertLessEqual((lm-x).square().mean().item(),(u-x).square().mean().item()+1e-6)
            if bits == 1:
                torch.testing.assert_close(lm,u)

    def test_degenerate(self):
        for x in (torch.zeros(7),torch.full((7,),2.),torch.tensor([-2.,3.])):
            for bits in (1,2):
                for method in ('optimized-uniform','lloyd-max'):
                    q,*_ = adaptive_quantize(x,bits,method)
                    self.assertTrue(torch.isfinite(q).all())
                    torch.testing.assert_close(q,x)

    def test_compression_output_really_controls_update(self):
        from experiments.adaptive_accuracy import compress_updates
        from compression.kashin_frame import FourierKashinFrame
        xs = torch.randn(3,8)
        frame = FourierKashinFrame(8,16)
        signs = torch.ones(16)
        def zero_quantizer(a,bits,method):
            return torch.zeros_like(a),torch.zeros_like(a,dtype=torch.long),torch.zeros(2),a.numel()*bits+64
        for method in ('SRK','Kashin'):
            with patch('experiments.adaptive_accuracy.adaptive_quantize',side_effect=zero_quantizer):
                update,_,_ = compress_updates(xs,method,1,'optimized-uniform',frame,signs,0,1)
            torch.testing.assert_close(update,torch.zeros(8))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
