"""Local reproducible quantizer study: generate / calibrate / evaluate / train.

Defaults are small validation runs. No remote work, publishing, or long run starts.
"""
import argparse
import copy
import hashlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from archive.compression.adaptive_quantization import adaptive_quantize
from archive.compression.quantizer_payload import METHODS, CodebookRegistry, canonical, encode, decode, rms32
from compression.srk import _fwht_fast
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve_balanced
from archive.experiments.matched_accuracy import load_data,minibatches,train_delta,evaluate
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import flatten_state_dict,unflatten_state_dict

D = 65536
REPRESENTATIONS = ('SRK','Kashin')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def environment():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(['git','ls-files','*.py'],cwd=root,text=True).splitlines()
    files = sorted(set(tracked+['compression/quantizer_payload.py','experiments/quantizer_study.py']))
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
        source_sha256={f:sha(root/f) for f in files if (root/f).exists()},
        git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-c','core.safecrlf=false','diff','HEAD'],cwd=root,stderr=subprocess.DEVNULL)).hexdigest(),
        python=platform.python_version(),torch=torch.__version__,hardware=platform.processor(),
        os=platform.platform(),device='cpu',threads=torch.get_num_threads(),
        accounting='ideal fixed-width packed uplink; no bit packing/networking implemented; excludes downlink/framing',
        normalization='aggregate squared error / mean client squared update norm')


def new_output(path,args):
    out = Path(path)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'metadata.json',dict(environment(),config=vars(args),status='running'))
    return out


def complete(out,**fields):
    metadata = json.loads((out/'metadata.json').read_text())
    metadata.update(status='complete',**fields)
    write_json(out/'metadata.json',metadata)


def transforms(seed,d):
    if d != sum(p.numel() for p in MNIST_MLP().parameters()) or d>D:
        raise ValueError('Model/configuration dimension mismatch')
    frame = FourierKashinFrame(d,D,seed=seed+4000)
    g = torch.Generator().manual_seed(seed+4000)
    signs = (torch.rand(D,generator=g)<.5).float()*2-1
    return frame,signs


@torch.no_grad()
def coefficients(xs,representation,frame,signs):
    if representation == 'SRK':
        return [_fwht_fast(F.pad(x,(0,len(signs)-len(x)))*signs) for x in xs]
    return [kashin_solve_balanced(frame,x,iterations=20) for x in xs]


def inverse(q,representation,frame,signs,d):
    return (_fwht_fast(q)*signs)[:d] if representation=='SRK' else frame.frame_synthesis(q)


@torch.no_grad()
def apply_update(model,update):
    before,shapes = flatten_state_dict(model.state_dict())
    if update.shape!=before.shape or not torch.isfinite(update).all():
        raise ValueError('Invalid server update')
    model.load_state_dict(unflatten_state_dict(before+update,shapes))


def load_inputs(path):
    root = Path(path)
    manifest = json.loads((root/'manifest.json').read_text())
    records = []
    for record in manifest['records']:
        file = root/record['file']
        if sha(file)!=record['sha256']:
            raise ValueError(f'Input hash mismatch: {file}')
        data = torch.load(file,map_location='cpu',weights_only=True)
        if data['seed']!=record['seed'] or data['round']!=record['round'] or data['clients']!=record['clients']:
            raise ValueError('Input provenance mismatch')
        if data['updates'].shape!=(3,manifest['d']) or not torch.isfinite(data['updates']).all():
            raise ValueError('Invalid saved updates')
        records.append((record,data['updates']))
    return manifest,records


def generate(args):
    out = new_output(args.out,args)
    x,y,_,_ = load_data()
    records=[]
    d = sum(p.numel() for p in MNIST_MLP().parameters())
    for seed in args.seeds:
        torch.manual_seed(seed)
        model=MNIST_MLP()
        rng=np.random.default_rng(seed+8700)
        for r in range(1,max(args.snapshots)+1):
            clients=rng.choice(10,3,replace=False).tolist()
            xs=torch.stack([train_delta(model,x,y,minibatches(seed,r,c)) for c in clients])
            if r in args.snapshots:
                name=f'updates_seed{seed}_round{r}.pt'
                torch.save(dict(seed=seed,round=r,clients=clients,updates=xs),out/name)
                records.append(dict(file=name,sha256=sha(out/name),seed=seed,round=r,clients=clients,
                                    transform_seed=seed+4000))
            apply_update(model,xs.mean(0))
        print(f'reference seed={seed} complete',flush=True)
    root=Path(__file__).resolve().parents[1]
    manifest=dict(records=records,d=d,D=D,source=environment(),
        trajectory='uncompressed FedAvg; independent training seeds',train_examples=6000,total_clients=10,
        participants=3,epochs=2,lr=.05,batch_size=32,
        data_sha256={name:sha(root/'data/MNIST/raw'/name) for name in
                     ('train-images-idx3-ubyte','train-labels-idx1-ubyte')})
    write_json(out/'manifest.json',manifest)
    complete(out,manifest_sha256=sha(out/'manifest.json'))


def reuse(args):
    """Register existing matched_distortion snapshots after verifying stored hashes."""
    import os
    out=new_output(args.out,args)
    source=Path(args.source)
    metadata=json.loads((source/'metadata.json').read_text())
    if metadata.get('status')!='complete' or 'input_sha256' not in metadata:
        raise ValueError('Existing run lacks a complete hashed input manifest')
    records=[];dimension=None
    for seed in args.seeds:
        for rnd in args.snapshots:
            name=f'updates_seed{seed}_round{rnd}.pt'
            file=source/name
            if metadata['input_sha256'].get(name)!=sha(file):
                raise ValueError(f'Historical input hash mismatch: {name}')
            data=torch.load(file,map_location='cpu',weights_only=True)
            if data['seed']!=seed or data['round']!=rnd or data['updates'].shape[0]!=3:
                raise ValueError('Historical input provenance mismatch')
            dimension=data['updates'].shape[1]
            records.append(dict(file=os.path.relpath(file,out),sha256=sha(file),seed=seed,round=rnd,
                clients=data['clients'],transform_seed=seed+4000))
    manifest=dict(records=records,d=dimension,D=D,source=metadata,
        historical_metadata_sha256=sha(source/'metadata.json'),
        trajectory='historical uncompressed FedAvg snapshots; no regenerated or overwritten inputs')
    write_json(out/'manifest.json',manifest)
    load_inputs(out)  # also checks model dimension, shapes and finite values
    complete(out,manifest_sha256=sha(out/'manifest.json'),historical_source_commit=metadata['commit'])


def calibrate(args):
    out=new_output(args.out,args)
    manifest,records=load_inputs(args.inputs)
    g=torch.Generator().manual_seed(args.sample_seed)
    pools={representation:[] for representation in REPRESENTATIONS}
    logs=[]
    started=time.perf_counter()
    for record,xs in records:
        frame,signs=transforms(record['seed'],xs.shape[1])
        # Same coordinate subset and equal weight per vector for both representations.
        ids=[torch.randperm(D,generator=g)[:min(D,args.samples_per_vector)] for x in xs]
        for representation in REPRESENTATIONS:
            start=time.perf_counter()
            aa=coefficients(xs,representation,frame,signs)
            transform_seconds=time.perf_counter()-start
            for client,(a,indices) in enumerate(zip(aa,ids)):
                s=rms32(a)
                if s==0:
                    raise ValueError('Zero calibration vector has no standardized distribution')
                pools[representation].append((a/s)[indices])
                logs.append(dict(seed=record['seed'],round=record['round'],client=record['clients'][client],
                    representation=representation,scale=float(s),sample_count=len(indices),
                    index_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                    transform_seconds_batch=transform_seconds))
    books={}
    fitting=[]
    for representation in REPRESENTATIONS:
        pool=torch.cat(pools[representation])
        for bits in (1,2):
            for method in ('fixed-uniform','fixed-lloyd-max'):
                if canonical(method,bits)!=method:
                    continue
                start=time.perf_counter(); diag={}
                fit_method='optimized-uniform' if method=='fixed-uniform' else 'lloyd-max'
                _,_,metadata,_=adaptive_quantize(pool,bits,fit_method,diagnostics=diag)
                centers=(metadata[0]+torch.arange(2**bits)*(metadata[-1]-metadata[0])/(2**bits-1)
                         if method=='fixed-uniform' else metadata)
                books[CodebookRegistry.key(representation,bits,method)]=centers
                fitting.append(dict(representation=representation,bits=bits,method=method,
                    samples=len(pool),seconds=time.perf_counter()-start,diagnostics=diag,centers=centers.tolist()))
    provenance=dict(input_manifest_sha256=sha(Path(args.inputs)/'manifest.json'),records=manifest['records'],
        source=environment(),sample_seed=args.sample_seed,samples_per_vector=args.samples_per_vector,
        weighting='equal count per complete vector; same sampled coordinates across representations',
        fitting=fitting,normalization='uncentered RMS; one float32 scale; no Gaussian assumption')
    registry=CodebookRegistry(books,provenance)
    registry.save(out/'codebooks.json')
    write_json(out/'calibration.json',logs)
    complete(out,offline_total_seconds=time.perf_counter()-started,shared_centers_bits=registry.shared_center_bits,
             codebook_sha256=registry.fingerprint())


def ensure_independent(registry,records):
    calibration_seeds={r['seed'] for r in registry.provenance['records']}
    if calibration_seeds & {r['seed'] for r in records}:
        raise ValueError('Calibration and evaluation/training seeds must be disjoint')


def timed(fn,repeats):
    fn()  # CPU warmup; returned warmup not used as measured trial
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter(); result=fn(); samples.append(time.perf_counter()-start)
    return result,dict(median_seconds=statistics.median(samples),
        iqr_seconds=float(np.quantile(samples,.75)-np.quantile(samples,.25)),raw_seconds=samples)


def evaluate_inputs(args):
    out=new_output(args.out,args)
    manifest,records=load_inputs(args.inputs)
    registry=CodebookRegistry.load(args.codebooks)
    ensure_independent(registry,[r for r,_ in records])
    frozen=registry.fingerprint()
    raw=[];payload_logs=[];transform_logs=[]
    for record,xs in records:
        seed,r=record['seed'],record['round']
        frame,signs=transforms(seed,xs.shape[1])
        mean_energy=xs.double().square().sum(1).mean().item()
        for representation in REPRESENTATIONS:
            aa,transform_time=timed(lambda:coefficients(xs,representation,frame,signs),args.timing_repeats)
            transform_logs.append(dict(seed=seed,round=r,representation=representation,**transform_time))
            for bits in (1,2):
                cache={}
                for method in METHODS:
                    key=canonical(method,bits)
                    if key!=method:
                        # Equivalent one-bit algorithms share actual fitted results/timing.
                        source_rows,source_logs=cache[key]
                        raw.extend(dict(row,method=method,alias_of=key) for row in source_rows)
                        payload_logs.extend(dict(row,method=method,alias_of=key) for row in source_logs)
                        continue
                    rows=[];logs=[]
                    draws=args.draws if method=='minmax-stochastic' else 1
                    for draw in range(draws):
                        recovered=[];per_client=[]
                        for client,(x,a) in enumerate(zip(xs,aa)):
                            rounding_seed=seed*1000000+r*10000+draw*10+client
                            def pack():
                                g=torch.Generator().manual_seed(rounding_seed)
                                return encode(a,bits,method,representation,registry,g)
                            payload,enc_time=timed(pack,args.timing_repeats)
                            q,dec_time=timed(lambda:decode(payload,registry),args.timing_repeats)
                            reconstructed=inverse(q,representation,frame,signs,len(x))
                            recovered.append(reconstructed)
                            mse=(q.double()-a.double()).square().mean().item()
                            per_client.append(dict(coefficient_mse=mse,
                                normalized_coefficient_mse=mse/max(a.double().square().mean().item(),1e-300),
                                client_nmse=(reconstructed.double()-x.double()).square().sum().item()/max(x.double().square().sum().item(),1e-300),
                                wire_bits=payload.wire_bits,encode_ms=1000*enc_time['median_seconds'],
                                decode_ms=1000*dec_time['median_seconds']))
                            logs.append(dict(seed=seed,round=r,representation=representation,bits=bits,method=method,
                                draw=draw,client=record['clients'][client],rounding_seed=rounding_seed,
                                metadata=payload.metadata.tolist(),diagnostics=payload.diagnostics,
                                encode_timing=enc_time,decode_timing=dec_time))
                        error=torch.stack(recovered).double().mean(0)-xs.double().mean(0)
                        rows.append(dict(seed=seed,round=r,representation=representation,bits=bits,method=method,
                            draw=draw,alias_of='',aggregate_nmse=error.square().sum().item()/max(mean_energy,1e-300),
                            **{k:float(np.mean([z[k] for z in per_client])) for k in per_client[0]}))
                    cache[key]=rows,logs
                    raw.extend(rows);payload_logs.extend(logs)
        print(f'evaluated seed={seed} round={r}',flush=True)
    if frozen!=registry.fingerprint():
        raise RuntimeError('Online quantization modified shared codebooks')
    write_json(out/'raw.json',raw);write_json(out/'payloads.json',payload_logs)
    write_json(out/'transforms.json',transform_logs)
    per_seed=[];summary=[];paired=[]
    metrics=('aggregate_nmse','coefficient_mse','normalized_coefficient_mse','client_nmse','wire_bits','encode_ms','decode_ms')
    for representation in REPRESENTATIONS:
        for bits in (1,2):
            for method in METHODS:
                points=[]
                for seed in sorted({r['seed'] for r in raw}):
                    # Average draws within a snapshot, then equal weight snapshots within seed.
                    snapshots=[]
                    for rnd in sorted({r['round'] for r in raw if r['seed']==seed}):
                        rr=[r for r in raw if (r['seed'],r['round'],r['representation'],r['bits'],r['method'])==(seed,rnd,representation,bits,method)]
                        snapshots.append({k:float(np.mean([r[k] for r in rr])) for k in metrics})
                    p=dict(seed=seed,representation=representation,bits=bits,method=method,
                           **{k:float(np.mean([r[k] for r in snapshots])) for k in metrics})
                    per_seed.append(p);points.append(p)
                summary.append(dict(representation=representation,bits=bits,method=method,seeds=len(points),
                    **{k:float(np.mean([r[k] for r in points])) for k in metrics},
                    nmse_seed_std=float(np.std([r['aggregate_nmse'] for r in points],ddof=1)) if len(points)>1 else 0))
            for source,target in (('minmax-stochastic','minmax-nearest'),('minmax-nearest','optimized-uniform'),
                                   ('optimized-uniform','lloyd-max'),('optimized-uniform','fixed-uniform'),('lloyd-max','fixed-lloyd-max')):
                differences=[]
                for seed in sorted({r['seed'] for r in raw}):
                    a=next(r for r in per_seed if (r['seed'],r['representation'],r['bits'],r['method'])==(seed,representation,bits,source))
                    b=next(r for r in per_seed if (r['seed'],r['representation'],r['bits'],r['method'])==(seed,representation,bits,target))
                    differences.append(dict(seed=seed,nmse_reduction=1-b['aggregate_nmse']/a['aggregate_nmse'],
                        coefficient_mse_reduction=1-b['coefficient_mse']/a['coefficient_mse'],
                        encode_time_reduction=1-b['encode_ms']/a['encode_ms']))
                paired.append(dict(representation=representation,bits=bits,source=source,target=target,per_seed=differences,
                    **{k:float(np.mean([r[k] for r in differences])) for k in differences[0] if k!='seed'}))
    write_json(out/'seed_summary.json',per_seed);write_json(out/'summary.json',summary);write_json(out/'paired.json',paired)
    complete(out,input_manifest_sha256=sha(Path(args.inputs)/'manifest.json'),codebook_sha256=frozen,
        shared_center_bits=registry.shared_center_bits,draws=args.draws,timing='CPU warmup; median and IQR; inverse transform excluded',
        deterministic_aliases='1-bit C=D, E=F; no independent repeated deterministic trials')


def aggregate_payload_updates(xs,representation,bits,method,frame,signs,registry,seed,r):
    aa=coefficients(xs,representation,frame,signs)
    recovered=[];logs=[]
    for client,(x,a) in enumerate(zip(xs,aa)):
        g=torch.Generator().manual_seed(seed*100000+r*100+client+9000000)
        payload=encode(a,bits,method,representation,registry,g)
        q=decode(payload,registry)
        recovered.append(inverse(q,representation,frame,signs,len(x)))
        logs.append(dict(client_slot=client,metadata=payload.metadata.tolist(),wire_bits=payload.wire_bits,
            diagnostics=payload.diagnostics))
    update=torch.stack(recovered).mean(0)
    nmse=((update.double()-xs.double().mean(0)).square().sum()/xs.double().square().sum(1).mean().clamp_min(1e-300)).item()
    return update,nmse,logs


def train(args):
    out=new_output(args.out,args)
    registry=CodebookRegistry.load(args.codebooks)
    ensure_independent(registry,[dict(seed=s) for s in args.seeds])
    frozen=registry.fingerprint()
    x,y,tx,ty=load_data()
    specs=[('Original',0,'none')]+[(rep,b,m) for b in (1,2) for rep in REPRESENTATIONS
                                for m in METHODS if canonical(m,b)==m]
    rows=[];logs=[]
    for seed in args.seeds:
        torch.manual_seed(seed)
        base=MNIST_MLP(); d=sum(p.numel() for p in base.parameters()); frame,signs=transforms(seed,d)
        models={spec:copy.deepcopy(base) for spec in specs}; budgets=dict.fromkeys(specs,0)
        rng=np.random.default_rng(seed+8700)
        for r in range(1,args.rounds+1):
            clients=rng.choice(10,3,replace=False).tolist();batches=[minibatches(seed,r,c) for c in clients]
            for spec in specs:
                representation,bits,method=spec;model=models[spec]
                xs=torch.stack([train_delta(model,x,y,batch) for batch in batches])
                if representation=='Original':
                    update,nmse,diagnostics=xs.mean(0),0.,[];used=d*32*3
                else:
                    update,nmse,diagnostics=aggregate_payload_updates(xs,representation,bits,method,frame,signs,registry,seed,r)
                    used=sum(z['wire_bits'] for z in diagnostics)
                apply_update(model,update)
                acc,loss=evaluate(model,tx,ty);budgets[spec]+=used
                rows.append(dict(seed=seed,round=r,representation=representation,bits=bits,method=method,
                    accuracy=acc,test_loss=loss,aggregate_nmse=nmse,total_uplink_bits=budgets[spec],
                    clients=clients,aliases=([k for k in METHODS if k!=method and canonical(k,bits)==method] if bits else [])))
                logs.extend(dict(seed=seed,round=r,representation=representation,bits=bits,method=method,**z) for z in diagnostics)
            write_json(out/'rounds.json',rows);write_json(out/'payloads.json',logs)
            print(f'train seed={seed} round={r}; {len(specs)} unique paths complete',flush=True)
    if frozen!=registry.fingerprint():
        raise RuntimeError('Training modified codebooks')
    complete(out,codebook_sha256=frozen,shared_center_bits=registry.shared_center_bits,
             warning='Separate model trajectories; NMSE not a same-input comparison. Short runs establish only pipeline execution.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=2)
    subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('generate');p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--snapshots',nargs='+',type=int,default=[1,2]);p.add_argument('--out',required=True)
    p=subs.add_parser('reuse');p.add_argument('--source',required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--snapshots',nargs='+',type=int,default=[1,15]);p.add_argument('--out',required=True)
    p=subs.add_parser('calibrate');p.add_argument('--inputs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sample-seed',type=int,default=2718);p.add_argument('--samples-per-vector',type=int,default=2048)
    p=subs.add_parser('evaluate');p.add_argument('--inputs',required=True);p.add_argument('--codebooks',required=True)
    p.add_argument('--draws',type=int,default=8);p.add_argument('--timing-repeats',type=int,default=3);p.add_argument('--out',required=True)
    p=subs.add_parser('train');p.add_argument('--codebooks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--rounds',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if Path(args.out).exists():
        parser.error('Output path already exists; choose a new path to preserve history')
    if args.threads<1 or any(getattr(args,k,1)<1 for k in ('draws','timing_repeats','samples_per_vector','rounds')):
        parser.error('Counts must be positive')
    if hasattr(args,'snapshots') and min(args.snapshots)<1:
        parser.error('Snapshot rounds must be positive')
    if hasattr(args,'seeds') and len(set(args.seeds))!=len(args.seeds):
        parser.error('Seeds must be unique')
    try:
        globals()[dict(generate='generate',reuse='reuse',calibrate='calibrate',evaluate='evaluate_inputs',train='train')[args.action]](args)
    except Exception as error:
        metadata_path=Path(args.out)/'metadata.json'
        if metadata_path.exists():
            metadata=json.loads(metadata_path.read_text())
            if metadata.get('status')=='running':
                metadata.update(status='failed',error=f'{type(error).__name__}: {error}')
                write_json(metadata_path,metadata)
        raise
    print(f'COMPLETE {args.out}',flush=True)


if __name__=='__main__':
    main()
