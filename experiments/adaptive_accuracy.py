"""Controlled 1/2-bit adaptive quantizer FL comparison; python -m experiments.adaptive_accuracy."""
import argparse
import copy
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.matched_accuracy import load_data, minibatches, train_delta, evaluate, write_csv
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import flatten_state_dict, unflatten_state_dict
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve_balanced
from compression.srk import _fwht_fast
from compression.quantization import stochastic_k_level_quantize
from compression.adaptive_quantization import adaptive_quantize


@torch.no_grad()
def compress_updates(xs, method, bits, quantizer, frame, signs, seed, round_index):
    recovered, diagnostics = [], []
    for client, x in enumerate(xs):
        started = time.perf_counter()
        a = (_fwht_fast(F.pad(x, (0, len(signs)-len(x)))*signs) if method == 'SRK'
             else kashin_solve_balanced(frame, x, iterations=20))
        transform_seconds = time.perf_counter()-started
        started = time.perf_counter()
        if quantizer == 'minmax-stochastic':
            generator = torch.Generator().manual_seed(seed*100000+round_index*100+client+9000000)
            q, _ = stochastic_k_level_quantize(a, 2**bits, generator)
            metadata = torch.stack([a.min(), a.max()])
            wire_bits = a.numel()*bits+64
        else:
            q, _, metadata, wire_bits = adaptive_quantize(a, bits, quantizer)
        quantizer_seconds = time.perf_counter()-started
        recovered.append((_fwht_fast(q)*signs)[:len(x)] if method == 'SRK' else frame.frame_synthesis(q))
        diagnostics.append(dict(client_slot=client,codebook=metadata.tolist(),
            coefficient_mse=(q-a).square().mean().item(),
            coefficient_mean_error=(q-a).mean().item(),
            # Coordinate mean error is NOT a measurement of statistical unbiasedness.
            client_nmse=((recovered[-1]-x).square().sum()/x.square().sum().clamp_min(1e-30)).item(),
            transform_seconds=transform_seconds,quantizer_seconds=quantizer_seconds,wire_bits=wire_bits))
    update = torch.stack(recovered).mean(0)
    error = update-xs.mean(0)
    nmse = (error.square().sum()/xs.square().sum(1).mean().clamp_min(1e-30)).item()
    return update, nmse, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    parser.add_argument('--rounds', type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    out = root/'results'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_adaptive_accuracy'
    out.mkdir(parents=True)
    metadata = dict(vars(args), commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status='running', D=65536, participants=3,total_clients=10,train_examples=6000,test_examples=10000,
        lr=.05,local_epochs=2,batch_size=32,kashin_solver='balanced-20; unchanged range objective',
        fitting='per client, per round, coefficient empirical MSE only; no test selection',
        accounting='ideal packed uplink; uniform 64 metadata bits; Lloyd-Max 32*2**bits; no measured networking',
        warning='adaptive quantizers are biased; local optimization, not certified global optimum')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'OUTPUT={out}',flush=True)
    x,y,tx,ty = load_data()
    specs = [('Original',0,'none')]+[(method,bits,q) for bits in (1,2)
        for method in ('SRK','Kashin') for q in ('minmax-stochastic','optimized-uniform','lloyd-max')]
    rows, diagnostic_rows = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        base = MNIST_MLP()
        flat, shapes = flatten_state_dict(base.state_dict())
        frame = FourierKashinFrame(len(flat),65536,seed=seed+4000)
        generator = torch.Generator().manual_seed(seed+4000)
        signs = (torch.rand(65536,generator=generator)<.5).float()*2-1
        models = {spec:copy.deepcopy(base) for spec in specs}
        budgets = dict.fromkeys(specs,0)
        rng = np.random.default_rng(seed+8700)
        for r in range(1,args.rounds+1):
            clients = rng.choice(10,3,replace=False).tolist()
            batches = [minibatches(seed,r,c) for c in clients]
            for spec in specs:
                method,bits,q = spec
                label = f'{method}-{bits}bit-{q}' if bits else 'Original'
                model = models[spec]
                deltas = torch.stack([train_delta(model,x,y,batch) for batch in batches])
                started = time.perf_counter()
                if method == 'Original':
                    update,nmse,diagnostics = deltas.mean(0),0.,[]
                    budgets[spec] += 3*len(flat)*32
                else:
                    update,nmse,diagnostics = compress_updates(deltas,method,bits,q,frame,signs,seed,r)
                    budgets[spec] += sum(z['wire_bits'] for z in diagnostics)
                seconds = time.perf_counter()-started
                before,_ = flatten_state_dict(model.state_dict())
                if not torch.isfinite(update).all():
                    raise RuntimeError(f'nonfinite update: {seed}, {r}, {label}')
                model.load_state_dict(unflatten_state_dict(before+update,shapes))
                acc,loss = evaluate(model,tx,ty)
                rows.append(dict(seed=seed,round=r,label=label,accuracy=acc,test_loss=loss,
                    aggregate_nmse=nmse,total_uplink_bits=budgets[spec],compression_seconds=seconds))
                diagnostic_rows.extend(dict(seed=seed,round=r,label=label,client_id=clients[z['client_slot']],**z) for z in diagnostics)
            write_csv(out/'rounds.csv',rows)
            # JSON preserves variable-length per-client codebooks exactly.
            (out/'diagnostics.json').write_text(json.dumps(diagnostic_rows),encoding='utf-8')
            print(f'seed={seed} round={r}: '+', '.join(f'{z["label"]}={100*z["accuracy"]:.2f}%' for z in rows[-len(specs):]),flush=True)
    metadata['status']='complete'
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'COMPLETE {out}',flush=True)


if __name__ == '__main__':
    main()
