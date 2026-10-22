import copy
import numpy as np
import pytest
import torch

from experiments.full_quantizer_training import SPECS, key, wire_bits, instrumented_aggregate
from experiments.summarize_quantizer_training import paired_summary, target, endpoint, validate, compare
from experiments import quantizer_study as old
from compression.quantizer_payload import CodebookRegistry


def rows(rounds=50):
    return [dict(seed=0, round=r, canonical_config=key(spec), representation=spec[0], bits=spec[1], method=spec[2],
                 clients=[0,1,2], status='complete', aliases=['lloyd-max'] if spec[1]==1 and spec[2]=='optimized-uniform' else
                 ['fixed-lloyd-max'] if spec[1]==1 and spec[2]=='fixed-uniform' else [], accuracy=.9, test_loss=.3,
                 total_uplink_bits=r*wire_bits(spec), round_uplink_bits=wire_bits(spec),
                 **dict.fromkeys(('local_training_seconds','compression_seconds','transform_seconds','encode_seconds',
                 'decode_seconds','inverse_seconds','aggregation_seconds','server_apply_seconds','evaluation_seconds',
                 'diagnostic_seconds','serial_algorithm_seconds','snapshot_io_seconds'), .01))
            for r in range(1,rounds+1) for spec in SPECS]


def test_alias_and_payload_validation():
    assert len(SPECS)==21
    assert validate(rows(), [0],50)['unique_rows']==1050
    with pytest.raises(ValueError,match='Duplicate'):
        validate(rows()+rows()[:1],[0],50)
    with pytest.raises(ValueError,match='Missing'):
        validate(rows()[1:],[0],50)
    broken=rows();broken[0]['total_uplink_bits']+=1
    with pytest.raises(ValueError,match='payload'):
        validate(broken,[0],50)
    broken=rows();broken[1]['clients']=[3,4,5]
    with pytest.raises(ValueError,match='pairing'):
        validate(broken,[0],50)


def test_endpoint_tail_units_pair_keys():
    rr=[r for r in rows() if r['representation']=='Original']
    for row in rr:
        row['accuracy']=row['round']/100
    ep=endpoint(rr)
    assert ep['accuracy']==.5
    assert ep['tail_accuracy']==pytest.approx(.48)
    eps=[]
    for s in range(5):
        for config,acc in [('new',.91+s*.001),('ref',.90+s*.001)]:
            eps.append(dict(seed=s,canonical_config=config,status='complete',accuracy=acc,tail_accuracy=acc,test_loss=.2))
    result=compare(list(reversed(eps)),'new','ref','test')
    assert result['mean_pp']==pytest.approx(1.)
    assert result['n']==5
    assert paired_summary([1])['ci_low_pp'] is None
    assert paired_summary([])['n']==0
    eps[0]['status']='execution_failed'
    assert compare(eps,'new','ref','test')['n']==4
    assert endpoint(rr[:-1])['status']=='execution_failed'


def test_thresholds_confirmation_and_failures():
    rr=[r for r in rows() if r['representation']=='Original']
    for row in rr:
        row['accuracy']=.8 if row['round']<3 else .91
    rr[3]['accuracy']=.8
    assert target(rr,.9)['round']==3
    assert target(rr,.9,True)['round']==7
    assert target(rr,.9,True)['uplink_bits']==rr[6]['total_uplink_bits']
    assert target(rr,.92)['status']=='not_reached'
    rr[1]['status']='diverged'
    assert target(rr,.9)['status']=='diverged'
    assert endpoint(rr)['status']=='diverged'
    rr[1]['status']='execution_failed'
    assert target(rr,.9)['status']=='execution_failed'


def test_invalid_alias_and_incomplete_target():
    broken=rows();broken[1]['aliases']=['made-up-alias']
    with pytest.raises(ValueError,match='Alias'):
        validate(broken,[0],50)
    rr=[r for r in rows(2) if r['representation']=='Original']
    assert target(rr,.92)['status']=='execution_failed'


def test_sample_sd_and_ci_are_seed_level():
    result=paired_summary([-2,-1,0,1,2])
    assert result['sample_sd_pp']==pytest.approx(np.sqrt(2.5))
    assert result['mean_pp']==0
    assert result['ci_low_pp']==pytest.approx(-1.963243161,rel=1e-6)


def test_wire_protocol_all_unique_paths():
    for rep in ('SRK','Kashin'):
        for b in (1,2):
            assert wire_bits((rep,b,'minmax-stochastic'))==3*(65536*b+64)
            assert wire_bits((rep,b,'fixed-uniform'))==3*(65536*b+32)
            assert wire_bits((rep,b,'lloyd-max'))==3*(65536*b+32*2**b)


def test_instrumentation_preserves_exact_update_and_rng():
    torch.set_num_threads(2)
    registry=CodebookRegistry.load('outputs/results/quantizer_study_20261004/calibration_v2/codebooks.json')
    frame,signs=old.transforms(0,50890)
    g=torch.Generator().manual_seed(2026)
    xs=torch.randn((3,50890),generator=g)*.001
    before=torch.random.get_rng_state().clone()
    for rep, bits, method in [s for s in SPECS if s[0]!='Original']:
        expected,nmse,_=old.aggregate_payload_updates(xs,rep,bits,method,frame,signs,registry,0,1)
        actual,observed,logs,times=instrumented_aggregate(xs,rep,bits,method,frame,signs,registry,0,1,True)
        assert torch.equal(actual,expected)
        assert observed==nmse
        assert sum(z['wire_bits'] for z in logs)==wire_bits((rep,bits,method))
        assert times['compression_seconds']>=sum(times[k] for k in ('transform_seconds','encode_seconds','decode_seconds','inverse_seconds','aggregation_seconds'))
    assert torch.equal(before,torch.random.get_rng_state())
