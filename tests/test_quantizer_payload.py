import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

from archive.compression.quantizer_payload import (METHODS,Payload,CodebookRegistry,canonical,
                                          encode,decode,rms32,uniform_centers)
from compression.quantization import stochastic_k_level_quantize


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def registry():
    books={}
    for rep in ('SRK','Kashin'):
        for bits in (1,2):
            for method in ('fixed-uniform','fixed-lloyd-max'):
                if canonical(method,bits)==method:
                    books[CodebookRegistry.key(rep,bits,method)]=torch.linspace(-1,1,2**bits)
    return CodebookRegistry(books,dict(test=True))


@pytest.mark.parametrize('bits',[1,2])
@pytest.mark.parametrize('method',METHODS)
def test_independent_payload_decode(bits,method):
    reg=registry()
    for values in (torch.zeros(11),torch.full((11,),2.),torch.tensor([-2.,3.]),
                   torch.tensor([0.,2e-9,5e-9]),torch.tensor([0.,1e-30,2e-30])):
        p=encode(values,bits,method,'SRK',reg,torch.Generator().manual_seed(7))
        # Clone only transmitted fields; no original coefficients available to decoder.
        independent=Payload(p.indices.clone(),p.metadata.clone(),bits,method,'SRK')
        assert torch.equal(decode(p,reg),decode(independent,reg))
        expected=32 if method.startswith('fixed-') else 32*2**bits if method=='lloyd-max' else 64
        assert p.wire_bits==len(values)*bits+expected
        assert torch.isfinite(decode(p,reg)).all()


def test_small_range_regression_and_dtype():
    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([0.,2e-9,5e-9],dtype=dtype)
        q,idx=stochastic_k_level_quantize(x,2,torch.Generator().manual_seed(2))
        assert q.dtype==dtype
        assert not torch.equal(q,x)
        expected=x.min()+idx.to(dtype)*(x.max()-x.min())
        torch.testing.assert_close(q,expected,rtol=1e-6,atol=0)
    x=torch.tensor([0.,1e-40,2e-40])
    assert rms32(x)>0
    for dtype in (torch.float16,torch.float32,torch.float64):
        p=encode(torch.tensor([-.2,.3],dtype=dtype),1,'minmax-nearest','SRK')
        assert p.metadata.dtype==torch.float32 and decode(p).dtype==torch.float32
    with pytest.raises(ValueError):
        encode(torch.tensor([1e-100],dtype=torch.float64),1,'fixed-uniform','SRK',registry())
    for bits in (1,2):
        x=torch.tensor([0.,2e-9,5e-9])
        actual,_=stochastic_k_level_quantize(x,2**bits,torch.Generator().manual_seed(77))
        p=encode(x,bits,'minmax-stochastic','SRK',generator=torch.Generator().manual_seed(77))
        assert torch.equal(actual,decode(p))
    p=encode(torch.tensor([-3e38,0.,3e38]),2,'minmax-nearest','SRK')
    assert torch.isfinite(decode(p)).all()


def test_diagnostics_serializable_and_original_fitting_results_preserved():
    import subprocess
    from archive.compression.adaptive_quantization import adaptive_quantize
    namespace={}
    exec(subprocess.check_output(['git','show','351411e:compression/adaptive_quantization.py'],text=True),namespace)
    torch.manual_seed(19)
    x=torch.randn(256)
    for bits in (1,2):
        for method in ('optimized-uniform','lloyd-max'):
            diag={}
            current=adaptive_quantize(x,bits,method,diagnostics=diag)
            old=namespace['adaptive_quantize'](x,bits,method)
            for a,b in zip(current[:3],old[:3]):
                assert torch.equal(a,b)
            json.dumps(diag)
            assert diag['starts']==6
            assert all(0<=n<=30 for n in diag['iterations_per_start'])


def test_stochastic_probability_with_sampling_error_tolerance():
    torch.manual_seed(5)
    # Each vector has fixed endpoints; 0.25 rounds to upper with probability 0.25.
    x=torch.tensor([0.,.25,1.]).repeat(60000,1)
    q,idx=stochastic_k_level_quantize(x,2,torch.Generator().manual_seed(91))
    standard_error=(.25*.75/len(x))**.5
    assert abs(float(q[:,1].mean())-.25)<6*standard_error
    assert torch.equal(q,idx.float())


def test_nearest_equal_spacing_and_onebit_alias():
    x=torch.linspace(-3,5,111)
    reg=registry()
    for bits in (1,2):
        for method in METHODS:
            p=encode(x,bits,method,'SRK',reg)
            if method=='minmax-stochastic':
                continue
            if method.startswith('fixed-'):
                centers=reg.centers('SRK',bits,method)*p.metadata[0]
            elif method=='lloyd-max':
                centers=p.metadata
            else:
                centers=uniform_centers(p.metadata,bits)
                gaps=centers[1:]-centers[:-1]
                torch.testing.assert_close(gaps,gaps[0].expand_as(gaps))
            torch.testing.assert_close((decode(p,reg)-x).abs(),(x[:,None]-centers).abs().min(1).values)
    for a,b in (('optimized-uniform','lloyd-max'),('fixed-uniform','fixed-lloyd-max')):
        torch.testing.assert_close(decode(encode(x,1,a,'SRK',reg),reg),decode(encode(x,1,b,'SRK',reg),reg))


def test_registry_save_load_hash_and_frozen_online_no_fitting():
    reg=registry();before=reg.fingerprint()
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'books.json';reg.save(path)
        loaded=CodebookRegistry.load(path)
        assert loaded.fingerprint()==before
        for method in ('fixed-uniform','fixed-lloyd-max'):
            with patch('archive.compression.quantizer_payload.adaptive_quantize',side_effect=AssertionError('online fitting')):
                p=encode(torch.randn(20),2,method,'SRK',loaded)
                torch.testing.assert_close(decode(p,loaded),decode(p,reg))
        assert loaded.fingerprint()==before
        centers=loaded.centers('SRK',1,'fixed-uniform');centers.zero_()
        assert loaded.fingerprint()==before
        path.write_text(path.read_text().replace('"test": true','"test": false'))
        with pytest.raises(ValueError):
            CodebookRegistry.load(path)


@pytest.mark.parametrize('bad',[torch.tensor([]),torch.tensor([float('nan')]),
                               torch.tensor([float('inf')]),torch.tensor([1,2])])
def test_invalid_values(bad):
    for method in METHODS:
        with pytest.raises(ValueError):
            encode(bad,1,method,'SRK',registry())
    with pytest.raises(ValueError):
        stochastic_k_level_quantize(bad,2)


def test_invalid_decode():
    p=encode(torch.tensor([-1.,1.]),1,'minmax-nearest','SRK')
    for bad in (torch.tensor([-1,0]),torch.tensor([0,2]),torch.tensor([0.,1.])):
        q=copy.deepcopy(p);q.indices=bad
        with pytest.raises(ValueError):
            decode(q)
    q=copy.deepcopy(p);q.metadata=q.metadata.double()
    with pytest.raises(ValueError):
        decode(q)
    for levels in (0,1,2.5):
        with pytest.raises(ValueError):
            stochastic_k_level_quantize(torch.ones(3),levels)


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('representation',['SRK','Kashin'])
def test_zero_decoded_update_controls_aggregation_and_server_model(method,representation):
    from archive.experiments.quantizer_study import aggregate_payload_updates,apply_update
    from compression.kashin_frame import FourierKashinFrame
    from models.mnist_mlp import MNIST_MLP
    xs=torch.randn(3,8)
    frame=FourierKashinFrame(8,16);signs=torch.ones(16)
    with patch('archive.experiments.quantizer_study.decode',side_effect=lambda p,r:torch.zeros_like(p.indices,dtype=torch.float32)):
        update,_,_=aggregate_payload_updates(xs,representation,1,method,frame,signs,registry(),0,1)
    torch.testing.assert_close(update,torch.zeros(8),rtol=0,atol=0)
    model=MNIST_MLP();before={k:v.clone() for k,v in model.state_dict().items()}
    apply_update(model,torch.zeros(sum(p.numel() for p in model.parameters())))
    for key,value in model.state_dict().items():
        assert torch.equal(value,before[key])


def test_split_and_input_hash_checks():
    from archive.experiments.quantizer_study import ensure_independent,load_inputs
    reg=CodebookRegistry({},dict(records=[dict(seed=11)]))
    with pytest.raises(ValueError):
        ensure_independent(reg,[dict(seed=11)])
    ensure_independent(reg,[dict(seed=12)])
    with tempfile.TemporaryDirectory() as temp:
        Path(temp,'updates.pt').write_bytes(b'invalid')
        Path(temp,'manifest.json').write_text('{"records":[{"file":"updates.pt","sha256":"wrong"}]}')
        with pytest.raises(ValueError,match='hash mismatch'):
            load_inputs(temp)
