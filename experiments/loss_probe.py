"""Exploratory common-checkpoint probe after the pre-specified FL run.

No tuning or method selection. Measure how a single compressed aggregate
changes test cross entropy at identical uncompressed reference checkpoints.
"""
import copy
import json
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.matched_accuracy import load_data, minibatches, train_delta, evaluate, write_csv, METHODS
from compression.srk import _fwht_fast
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve, kashin_solve_balanced
from compression.quantization import stochastic_k_level_quantize
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import flatten_state_dict, unflatten_state_dict


def main():
    torch.set_num_threads(2)
    out=Path('results')/f'{datetime.now():%Y%m%d_%H%M%S}_loss_probe'; out.mkdir()
    meta=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        seeds=list(range(5)),snapshots=[1,15,50],trials=8,status='running',
        purpose='exploratory same-checkpoint immediate loss effect; no tuning; all configurations retained')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'OUTPUT={out.resolve()}',flush=True)
    x,y,tx,ty=load_data(); rows=[]
    for seed in range(5):
        torch.manual_seed(seed); model=MNIST_MLP(); flat,shapes=flatten_state_dict(model.state_dict()); d=len(flat)
        rng=np.random.default_rng(seed+8700)
        frame=FourierKashinFrame(d,65536,seed=seed+4000)
        g=torch.Generator().manual_seed(seed+4000); signs=(torch.rand(65536,generator=g)<.5).float()*2-1
        for r in range(1,51):
            ids=rng.choice(10,3,replace=False).tolist()
            xs=torch.stack([train_delta(model,x,y,minibatches(seed,r,c)) for c in ids])
            flat,_=flatten_state_dict(model.state_dict()); true=xs.mean(0)
            if r in meta['snapshots']:
                ref=copy.deepcopy(model); ref.load_state_dict(unflatten_state_dict(flat+true,shapes))
                ref_acc,ref_loss=evaluate(ref,tx,ty)
                coeff={}
                coeff['SRK']=[_fwht_fast(F.pad(v,(0,65536-d))*signs) for v in xs]
                coeff['Fourier-direct']=[frame.frame_analysis(v)/frame.A for v in xs]
                coeff['Kashin-legacy']=[kashin_solve(frame,v) for v in xs]
                coeff['Kashin-balanced']=[kashin_solve_balanced(frame,v) for v in xs]
                for bits in (1,2):
                    for method in METHODS:
                        for trial in range(8):
                            recovered=[]
                            for i,a in enumerate(coeff[method]):
                                g=torch.Generator().manual_seed(seed*1000000+r*10000+trial*10+i+7000000)
                                q,_=stochastic_k_level_quantize(a,2**bits,generator=g)
                                recovered.append((_fwht_fast(q)*signs)[:d] if method=='SRK' else frame.frame_synthesis(q))
                            update=torch.stack(recovered).mean(0)
                            test=copy.deepcopy(model); test.load_state_dict(unflatten_state_dict(flat+update,shapes))
                            acc,loss=evaluate(test,tx,ty)
                            rows.append(dict(seed=seed,snapshot=r,bits=bits,method=method,trial=trial,
                                reference_accuracy=ref_acc,reference_loss=ref_loss,accuracy=acc,loss=loss,
                                excess_loss=loss-ref_loss,delta_accuracy_pp=(acc-ref_acc)*100,
                                aggregate_nmse=float((update-true).square().sum()/xs.square().sum(1).mean())))
                write_csv(out/'raw.csv',rows)
                print(f'probe seed={seed} snapshot={r}',flush=True)
            model.load_state_dict(unflatten_state_dict(flat+true,shapes))
    summary=[]
    for r in meta['snapshots']:
        for bits in (1,2):
            for method in METHODS:
                subset=[z for z in rows if z['snapshot']==r and z['bits']==bits and z['method']==method]
                summary.append(dict(snapshot=r,bits=bits,method=method,
                    excess_loss=np.mean([z['excess_loss'] for z in subset]),
                    delta_accuracy_pp=np.mean([z['delta_accuracy_pp'] for z in subset]),
                    aggregate_nmse=np.mean([z['aggregate_nmse'] for z in subset])))
    write_csv(out/'summary.csv',summary)
    meta['status']='complete'; (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'COMPLETE {out.resolve()}',flush=True)


if __name__=='__main__': main()
