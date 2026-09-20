"""Same-input, exact-payload compression benchmark; no test-accuracy selection.

Run: python -u -m experiments.matched_distortion
All methods compress stored updates from one uncompressed reference trajectory.
Five independent training/frame seeds are inference units; rounding draws and
snapshots within a seed are repeated measurements, not independent seeds.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compression.srk import _fwht_fast
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve, kashin_solve_balanced
from compression.quantization import stochastic_k_level_quantize
from data.mnist_federated import build_mnist_federated_data
from federated.local_training import local_train_delta
from federated.aggregation import federated_round_original_update
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import flatten_state_dict
from utils.plot_style import configure_chinese_plotting

configure_chinese_plotting()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def reference_updates(seed, out, snapshots):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loaders, _ = build_mnist_federated_data(num_clients=10, images_per_client=600, paired_shuffle=True)
    model = MNIST_MLP()
    rng = np.random.default_rng(seed + 8700)
    saved = []
    for r in range(1, max(snapshots) + 1):
        ids = rng.choice(10, 3, replace=False).tolist()
        deltas = []
        for client in ids:
            loaders[client].generator.manual_seed(seed * 100000 + r * 100 + client)
            deltas.append(local_train_delta(model, loaders[client], epochs=2, lr=.05))
        if r in snapshots:
            xs = torch.stack([flatten_state_dict(delta)[0] for delta in deltas])
            torch.save({'updates': xs, 'clients': ids, 'seed': seed, 'round': r},
                       out / f'updates_seed{seed}_round{r}.pt')
            saved.append((r, xs, ids))
        federated_round_original_update(model, deltas)
    print(f'reference seed={seed}: snapshots={snapshots}, total data=6000, participants=3', flush=True)
    return saved


def configurations(d):
    D0 = 2 ** int(np.ceil(np.log2(d)))
    result = []
    for budget_bits in (1, 2, 3, 4):
        payload = D0 * budget_bits + 64
        result.append((budget_bits, 'SRK', budget_bits, D0, payload))
        # Matched dimensions/bit widths isolate transform and solver; alternate
        # widths test the redundancy-vs-precision tradeoff under the same cap.
        for bits in range(1, budget_bits + 1):
            D = (payload - 64) // bits
            for method in ('Fourier-direct', 'Kashin-legacy', 'Kashin-balanced'):
                result.append((budget_bits, method, bits, D, payload))
    return result


def compress_snapshot(seed, r, xs, trials, rows, diagnostics):
    d = xs.shape[1]
    target = xs.mean(0)
    mean_energy = xs.square().sum(1).mean().item()
    target_energy = target.square().sum().item()
    # Reuse a fixed transform across bit widths and across methods at each D.
    prepared = {}
    for budget, method, bits, D, cap in configurations(d):
        key = (method, D)
        if key not in prepared:
            start = time.perf_counter()
            if method == 'SRK':
                g = torch.Generator().manual_seed(seed + 4000)
                signs = (torch.rand(D, generator=g) < .5).float() * 2 - 1
                def decode(a, signs=signs):
                    return (_fwht_fast(a) * signs)[:d]
                coefficients = [_fwht_fast(torch.nn.functional.pad(x, (0, D-d)) * signs) for x in xs]
            else:
                frame = FourierKashinFrame(d, D, seed=seed + 4000)
                decode = frame.frame_synthesis
                if method == 'Fourier-direct':
                    coefficients = [frame.frame_analysis(x) / frame.A for x in xs]
                elif method == 'Kashin-legacy':
                    coefficients = [kashin_solve(frame, x, iterations=10) for x in xs]
                else:
                    coefficients = [kashin_solve_balanced(frame, x, iterations=20) for x in xs]
            elapsed = (time.perf_counter() - start) / len(xs)
            prepared[key] = coefficients, decode
            residuals = torch.stack([decode(a) - x for a, x in zip(coefficients, xs)])
            diagnostics.append(dict(seed=seed, snapshot=r, method=method, D=D, lambda_value=D/d,
                range_mean=float(np.mean([(a.max()-a.min()).item() for a in coefficients])),
                peak_mean=float(np.mean([a.abs().max().item() for a in coefficients])),
                reconstruction_relerr2=residuals.square().sum(1).mean().item()/mean_energy,
                aggregate_bias_nmse=residuals.mean(0).square().sum().item()/mean_energy,
                encode_seconds_per_client=elapsed))
        coefficients, decode = prepared[key]
        for trial in range(trials):
            errors = []
            for client, (x, a) in enumerate(zip(xs, coefficients)):
                # Independent across clients/draws, repeatable across methods.
                g = torch.Generator().manual_seed(seed*1000000 + r*10000 + trial*10 + client)
                q, _ = stochastic_k_level_quantize(a, 2**bits, generator=g)
                errors.append(decode(q)-x)
            errors = torch.stack(errors)
            aggregate_error = errors.mean(0).square().sum().item()
            rows.append(dict(seed=seed, snapshot=r, trial=trial, method=method, bits=bits,
                D=D, lambda_value=D/d, budget_id=budget, budget_bits_client=cap,
                used_bits_client=D*bits+64, participants=len(xs),
                aggregate_nmse=aggregate_error/mean_energy,
                aggregate_relative_error=aggregate_error/max(target_energy, 1e-30),
                client_nmse=errors.square().sum(1).mean().item()/mean_energy))
    print(f'compressed seed={seed}, snapshot={r}, trials={trials}', flush=True)


def summarize(rows, out):
    keys = sorted({(x['budget_id'], x['method'], x['bits'], x['D']) for x in rows})
    seeds = sorted({x['seed'] for x in rows})
    summary = []
    for budget, method, bits, D in keys:
        per_seed, ratios = [], []
        for seed in seeds:
            values = [x['aggregate_nmse'] for x in rows if (x['seed'], x['budget_id'], x['method'], x['bits'], x['D']) == (seed,budget,method,bits,D)]
            ref = [x['aggregate_nmse'] for x in rows if x['seed']==seed and x['budget_id']==budget and x['method']=='SRK']
            per_seed.append(float(np.mean(values)))
            ratios.append(float(np.mean(values)/np.mean(ref)))
        # Paired seed bootstrap; descriptive at n=5, never pool trials as seeds.
        rng = np.random.default_rng(829)
        boots = np.array(ratios)[rng.integers(0,len(seeds),(10000,len(seeds)))].mean(1)
        summary.append(dict(budget_id=budget, method=method, bits=bits, D=D,
            nmse_mean=float(np.mean(per_seed)), nmse_seed_std=float(np.std(per_seed,ddof=1)) if len(seeds)>1 else 0,
            ratio_to_srk=float(np.mean(ratios)), ratio_ci_low=float(np.quantile(boots,.025)),
            ratio_ci_high=float(np.quantile(boots,.975)), seeds=len(seeds)))
    write_csv(out/'summary.csv', summary)
    return summary


def plots(summary, out):
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    styles = {'SRK':('black','o'), 'Fourier-direct':('tab:gray','s'),
              'Kashin-legacy':('tab:blue','^'), 'Kashin-balanced':('tab:orange','D')}
    for method, (color, marker) in styles.items():
        values = sorted([x for x in summary if x['method']==method and x['D']==65536],key=lambda x:x['budget_id'])
        axes[0].plot([x['budget_id']*65536/50890+64/50890 for x in values],
                     [x['nmse_mean'] for x in values],marker=marker,color=color,label=method)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('上行通信量（bit/原始维度/客户端）')
    axes[0].set_ylabel('聚合更新失真（NMSE，越低越好）')
    axes[0].set_title('系数个数和 bit 宽度相同')
    axes[0].legend(fontsize=8)
    candidates = sorted([x for x in summary if x['method']=='Kashin-balanced'],key=lambda x:(x['budget_id'],x['bits']))
    positions = np.arange(len(candidates))
    means = np.array([x['ratio_to_srk'] for x in candidates])
    axes[1].errorbar(positions,means,yerr=[means-np.array([x['ratio_ci_low'] for x in candidates]),np.array([x['ratio_ci_high'] for x in candidates])-means],fmt='o',capsize=3,color='tab:orange')
    axes[1].axhline(1,color='black',ls='--',label='同预算 SRK')
    axes[1].set_yscale('log')
    axes[1].set_xticks(positions,[f'{x["budget_id"]}/{x["bits"]}\n{x["D"]/50890:.2f}' for x in candidates],fontsize=8)
    axes[1].set_xlabel('SRK bit / Kashin bit；第二行：冗余比例')
    axes[1].set_ylabel('Kashin 失真 / 同预算 SRK 失真（<1 更好）')
    axes[1].set_title('所有预先设定的同预算组合')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle('相同真实更新，5 个种子；不按准确率挑选配置')
    fig.tight_layout()
    fig.savefig(out/'matched_distortion.png',dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    parser.add_argument('--trials',type=int,default=12)
    parser.add_argument('--snapshots',nargs='+',type=int,default=[1,5,15])
    parser.add_argument('--tag',default='matched_distortion')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = root/'results'/f'{stamp}_{args.tag}'
    out.mkdir(parents=True,exist_ok=False)
    meta = dict(vars(args), commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        device='cpu',threads=2,torch=torch.__version__,total_clients=10,participating_clients=3,
        images_per_client=600,normalizer='mean client squared update norm',metadata_bits=64,
        accounting='ideal packed uplink payload; 2 float32 endpoints; shared transform seeds; excludes downlink and transport',
        sampling='common uncompressed training trajectory; mean over snapshots and quantization draws within each seed',
        solver_variant='shrinking clipping, exact residual correction, retain minimum-range feasible candidate')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'OUTPUT={out}',flush=True)
    rows, diagnostics = [], []
    for seed in args.seeds:
        for r, xs, ids in reference_updates(seed,out,args.snapshots):
            compress_snapshot(seed,r,xs,args.trials,rows,diagnostics)
        write_csv(out/'raw.csv',rows)
        write_csv(out/'solver_diagnostics.csv',diagnostics)
    summary=summarize(rows,out)
    plots(summary,out)
    # Hash preserved inputs so future reruns can verify identical updates.
    meta['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('updates_*.pt')}
    meta['status']='complete'
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for x in summary:
        if x['method']=='Kashin-balanced':
            print(f'budget={x["budget_id"]} Kashin bits={x["bits"]} lambda={x["D"]/50890:.3f}: ratio={x["ratio_to_srk"]:.4f} CI=[{x["ratio_ci_low"]:.4f},{x["ratio_ci_high"]:.4f}]',flush=True)
    print(f'COMPLETE {out}',flush=True)


if __name__=='__main__':
    main()
