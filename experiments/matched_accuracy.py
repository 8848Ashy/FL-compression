"""Pre-specified paired FL validation following matched_distortion.

Same 6000 training images, 10 clients, 3 sampled participants per round,
identical initialization, minibatch permutations and per-client rounding seeds.
No test-based model/configuration selection. No frontier maximization.
"""
import argparse
import copy
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

from compression.srk import _fwht_fast
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve, kashin_solve_balanced
from compression.quantization import stochastic_k_level_quantize
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import flatten_state_dict, unflatten_state_dict
from utils.plot_style import configure_chinese_plotting

configure_chinese_plotting()


METHODS = ('SRK', 'Fourier-direct', 'Kashin-legacy', 'Kashin-balanced')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_data():
    train=MNIST('./data',train=True,download=False)
    test=MNIST('./data',train=False,download=False)
    def images(ds,n):
        return (ds.data[:n].float().unsqueeze(1)/255-.1307)/.3081
    return images(train,6000),train.targets[:6000],images(test,10000),test.targets[:10000]


def minibatches(seed,r,client):
    generator=torch.Generator().manual_seed(seed*100000+r*100+client)
    return [chunk+client*600 for _ in range(2)
            for chunk in torch.randperm(600,generator=generator).split(32)]


def train_delta(model,x,y,batches):
    local=copy.deepcopy(model)
    local.train()
    optimizer=torch.optim.SGD(local.parameters(),lr=.05)
    for indices in batches:
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(local(x[indices]),y[indices]).backward()
        optimizer.step()
    return flatten_state_dict(local.state_dict())[0]-flatten_state_dict(model.state_dict())[0]


@torch.no_grad()
def evaluate(model,x,y):
    model.eval(); correct=0; loss=0.
    for start in range(0,len(y),512):
        logits=model(x[start:start+512]); labels=y[start:start+512]
        loss+=F.cross_entropy(logits,labels,reduction='sum').item()
        correct+=(logits.argmax(1)==labels).sum().item()
    return correct/len(y),loss/len(y)


@torch.no_grad()
def compress(xs,method,bits,frame,signs,seed,r):
    d=xs.shape[1]; errors=[]; recovered=[]
    for i,x in enumerate(xs):
        if method=='SRK':
            a=_fwht_fast(F.pad(x,(0,len(signs)-d))*signs)
        elif method=='Fourier-direct': a=frame.frame_analysis(x)/frame.A
        elif method=='Kashin-legacy': a=kashin_solve(frame,x,iterations=10)
        else: a=kashin_solve_balanced(frame,x,iterations=20)
        generator=torch.Generator().manual_seed(seed*100000+r*100+i+9000000)
        q,_=stochastic_k_level_quantize(a,2**bits,generator=generator)
        xhat=(_fwht_fast(q)*signs)[:d] if method=='SRK' else frame.frame_synthesis(q)
        recovered.append(xhat); errors.append(xhat-x)
    true=xs.mean(0); error=torch.stack(errors).mean(0)
    mean_energy=xs.square().sum(1).mean().clamp_min(1e-30)
    relative=error.square().sum()/true.square().sum().clamp_min(1e-30)
    return torch.stack(recovered).mean(0),float(error.square().sum()/mean_energy),float(relative)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--rounds',type=int,default=50)
    args=p.parse_args(); torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    out=root/'results'/f'{datetime.now():%Y%m%d_%H%M%S}_matched_accuracy'
    out.mkdir(parents=True,exist_ok=False)
    specs=[('Original','Original',32)]+[(f'{m}-{b}bit',m,b) for b in (1,2) for m in METHODS]
    metadata=dict(vars(args),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        total_clients=10,participants=3,training_examples=6000,test_examples=10000,device='cpu',
        lr=.05,epochs=2,batch_size=32,D=65536,lambda_value=65536/50890,
        specs=specs,targets=[.85,.90,.92],main_endpoint='round 50; paired seed differences',
        secondary_endpoint='mean of final 5 rounds; test cross entropy; uplink to fixed targets',
        accounting='D*bits+64 bits per participant per round; Original d*32; uplink only',
        randomization='same init, participation, explicit minibatches, shared frame/signs per seed; independent rounding across clients',
        status='running')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'OUTPUT={out}',flush=True)
    x,y,tx,ty=load_data(); rows=[]; endpoints=[]; target_rows=[]
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        base=MNIST_MLP(); flat,shapes=flatten_state_dict(base.state_dict()); d=len(flat)
        models={label:copy.deepcopy(base) for label,_,_ in specs}
        frame=FourierKashinFrame(d,65536,seed=seed+4000)
        g=torch.Generator().manual_seed(seed+4000)
        signs=(torch.rand(65536,generator=g)<.5).float()*2-1
        rng=np.random.default_rng(seed+8700)
        init_acc,init_loss=evaluate(base,tx,ty)
        print(f'seed={seed} initial accuracy={init_acc:.4f}',flush=True)
        for r in range(1,args.rounds+1):
            clients=rng.choice(10,3,replace=False).tolist()
            batches=[minibatches(seed,r,c) for c in clients]
            for label,method,bits in specs:
                model=models[label]; started=time.perf_counter()
                deltas=torch.stack([train_delta(model,x,y,bs) for bs in batches])
                train_seconds=time.perf_counter()-started; started=time.perf_counter()
                if method=='Original': update=deltas.mean(0); nmse=relative=0.
                else: update,nmse,relative=compress(deltas,method,bits,frame,signs,seed,r)
                compression_seconds=time.perf_counter()-started
                flat,_=flatten_state_dict(model.state_dict())
                if not torch.isfinite(update).all(): raise RuntimeError(f'nonfinite update {seed} {r} {label}')
                model.load_state_dict(unflatten_state_dict(flat+update,shapes))
                acc,loss=evaluate(model,tx,ty)
                payload=d*32 if method=='Original' else 65536*bits+64
                rows.append(dict(seed=seed,round=r,label=label,method=method,bits=bits,
                    accuracy=acc,test_loss=loss,aggregate_nmse=nmse,aggregate_relative_error=relative,
                    total_uplink_bits=payload*3*r,clients='-'.join(map(str,clients)),
                    training_seconds=train_seconds,compression_seconds=compression_seconds))
            if r%5==0:
                write_csv(out/'rounds.csv',rows)
                print(f'seed={seed} round={r}/{args.rounds}: '+', '.join(f'{a["label"]}={a["accuracy"]*100:.2f}' for a in rows[-len(specs):]),flush=True)
        for label,method,bits in specs:
            series=[z for z in rows if z['seed']==seed and z['label']==label]
            train_acc,train_loss=evaluate(models[label],x,y)
            endpoints.append(dict(seed=seed,label=label,method=method,bits=bits,
                final_accuracy=series[-1]['accuracy'],tail_accuracy=float(np.mean([z['accuracy'] for z in series[-5:]])),
                final_test_loss=series[-1]['test_loss'],train_accuracy=train_acc,train_loss=train_loss,
                mean_nmse=float(np.mean([z['aggregate_nmse'] for z in series]))))
            for target in metadata['targets']:
                eligible=[z for z in series if z['accuracy']>=target]
                point=eligible[0] if eligible else None
                target_rows.append(dict(seed=seed,label=label,target=target,
                    round=point['round'] if point else '',uplink_bits=point['total_uplink_bits'] if point else '',
                    status='reached' if point else 'not_reached'))
        write_csv(out/'endpoints.csv',endpoints); write_csv(out/'targets.csv',target_rows)
    summary=[]
    for label,method,bits in specs:
        points=[z for z in endpoints if z['label']==label]
        acc=np.array([z['final_accuracy']*100 for z in points])
        summary.append(dict(label=label,mean_accuracy=acc.mean(),std_accuracy=acc.std(ddof=1) if len(acc)>1 else 0,
            mean_tail_accuracy=np.mean([z['tail_accuracy']*100 for z in points]),
            mean_test_loss=np.mean([z['final_test_loss'] for z in points]),
            mean_train_accuracy=np.mean([z['train_accuracy']*100 for z in points]),
            mean_nmse=np.mean([z['mean_nmse'] for z in points])))
    write_csv(out/'summary.csv',summary)
    paired=[]
    for bits in (1,2):
        for method in METHODS[1:]:
            differences=[]; tail=[]; loss=[]
            for seed in args.seeds:
                a=next(z for z in endpoints if z['seed']==seed and z['label']==f'{method}-{bits}bit')
                b=next(z for z in endpoints if z['seed']==seed and z['label']==f'SRK-{bits}bit')
                differences.append((a['final_accuracy']-b['final_accuracy'])*100)
                tail.append((a['tail_accuracy']-b['tail_accuracy'])*100)
                loss.append(a['final_test_loss']-b['final_test_loss'])
            # t critical value for default five paired seeds (df=4).
            half=2.776445105*np.std(differences,ddof=1)/np.sqrt(5) if len(differences)==5 else float('nan')
            paired.append(dict(method=method,bits=bits,delta_accuracy_pp=np.mean(differences),
                ci_low=np.mean(differences)-half,ci_high=np.mean(differences)+half,
                seed_deltas=';'.join(f'{v:.4f}' for v in differences),delta_tail_pp=np.mean(tail),
                delta_test_loss=np.mean(loss),n=len(differences)))
    write_csv(out/'paired.csv',paired)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    palette={'Original':'black','SRK':'tab:red','Fourier-direct':'tab:gray','Kashin-legacy':'tab:blue','Kashin-balanced':'tab:orange'}
    for ax,bits in zip(axes,(1,2)):
        for method in ('Original',)+METHODS:
            label='Original' if method=='Original' else f'{method}-{bits}bit'
            means=[]; spreads=[]
            for r in range(1,args.rounds+1):
                values=[z['accuracy']*100 for z in rows if z['label']==label and z['round']==r]
                means.append(np.mean(values)); spreads.append(np.std(values,ddof=1) if len(values)>1 else 0)
            rr=np.arange(1,args.rounds+1); means=np.array(means); spreads=np.array(spreads)
            ax.plot(rr,means,label=method,color=palette[method])
            ax.fill_between(rr,means-spreads,means+spreads,alpha=.10,color=palette[method])
        ax.set_title(f'{bits}-bit，每轮压缩通信量相同')
        ax.set_xlabel('轮次（压缩方法在每轮的累计上行通信量相同）')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('测试准确率（%）'); axes[1].legend(fontsize=8)
    fig.suptitle(f'6000 张训练图片，每轮从 10 个客户端中选 3 个；{len(args.seeds)} 个种子，均值 ± 标准差')
    fig.tight_layout(); fig.savefig(out/'accuracy.png',dpi=200); plt.close(fig)
    metadata['status']='complete'
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('PAIRED '+json.dumps(paired,indent=2),flush=True)
    print(f'COMPLETE {out}',flush=True)


if __name__=='__main__': main()
