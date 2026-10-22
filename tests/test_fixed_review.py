import copy
import pytest
import torch
from experiments import full_quantizer_training as ft
from experiments import quantizer_study as old
from compression.quantizer_payload import CodebookRegistry


@pytest.mark.parametrize('rep,bits,method',[(rep,b,m) for rep in ('SRK','Kashin') for b,m in
                         ((1,'fixed-uniform'),(2,'fixed-uniform'),(2,'fixed-lloyd-max'))])
def test_zero_decoded_update_cannot_change_server(monkeypatch,rep,bits,method):
    torch.set_num_threads(2)
    torch.manual_seed(33)
    model=old.MNIST_MLP()
    before=old.flatten_state_dict(model.state_dict())[0].clone()
    baseline=copy.deepcopy(model)
    xs=torch.randn((3,50890),generator=torch.Generator().manual_seed(71))*.001
    frame,signs=old.transforms(0,50890)
    registry=CodebookRegistry.load('outputs/results/quantizer_full_20261008_01/assets/codebooks.json')
    original_decode=ft.decode
    calls=[]
    def zero_decode(payload,shared):
        q=original_decode(payload,shared)
        calls.append(payload.method)
        return torch.zeros_like(q)
    monkeypatch.setattr(ft,'decode',zero_decode)
    update,_,_,_=ft.instrumented_aggregate(xs,rep,bits,method,frame,signs,registry,0,1)
    old.apply_update(model,update)
    old.apply_update(baseline,xs.mean(0))
    assert len(calls)==3
    assert torch.count_nonzero(update)==0
    assert torch.equal(before,old.flatten_state_dict(model.state_dict())[0])
    assert not torch.equal(before,old.flatten_state_dict(baseline.state_dict())[0])
