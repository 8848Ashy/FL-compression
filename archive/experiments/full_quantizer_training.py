"""Instrumented continuation of quantizer_study; numerical algorithms are reused."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from archive.experiments import quantizer_study as old
from archive.compression.quantizer_payload import METHODS, canonical, encode, decode, rms32, CodebookRegistry
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import flatten_state_dict

FINGERPRINT = '1b200195cd0a5cad660f7f7a7344c66d0c68c082ab84c0ca3611ccd49218d5e5'
STAGES = (1, 15, 30, 50)
SPECS = [('Original', 0, 'none')] + [(rep, b, m) for b in (1, 2)
         for rep in old.REPRESENTATIONS for m in METHODS if canonical(m, b) == m]


def key(spec):
    return ':'.join(map(str, spec))


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def wire_bits(spec):
    rep, b, m = spec
    if rep == 'Original':
        return 50890 * 32 * 3
    metadata = (2**b if m == 'lloyd-max' else 1 if m.startswith('fixed-') else 2)
    return 3 * (65536 * b + 32 * metadata)


def instrumented_aggregate(xs, rep, bits, method, frame, signs, registry, seed, rnd, stage=False):
    """Direct compression clock excludes diagnostics and includes stack/mean overhead.

    Exclusive component clocks: transform, encode, decode, inverse, aggregation.
    Existing encoders may construct a discarded temporary reconstruction internally;
    that work remains in encode, without calling the external decoder twice.
    """
    recovered, payloads, qs = [], [], []
    timings = dict.fromkeys(('transform_seconds', 'encode_seconds', 'decode_seconds',
                            'inverse_seconds', 'aggregation_seconds'), 0.)
    start = time.perf_counter()
    tick = time.perf_counter()
    aa = old.coefficients(xs, rep, frame, signs)
    timings['transform_seconds'] = time.perf_counter() - tick
    for slot, (x, a) in enumerate(zip(xs, aa)):
        g = torch.Generator().manual_seed(seed*100000 + rnd*100 + slot + 9000000)
        tick = time.perf_counter()
        payload = encode(a, bits, method, rep, registry, g)
        timings['encode_seconds'] += time.perf_counter() - tick
        tick = time.perf_counter()
        q = decode(payload, registry)
        timings['decode_seconds'] += time.perf_counter() - tick
        tick = time.perf_counter()
        recovered.append(old.inverse(q, rep, frame, signs, len(x)))
        timings['inverse_seconds'] += time.perf_counter() - tick
        payloads.append(payload)
        qs.append(q)
    tick = time.perf_counter()
    update = torch.stack(recovered).mean(0)
    timings['aggregation_seconds'] = time.perf_counter() - tick
    timings['compression_seconds'] = time.perf_counter() - start
    tick = time.perf_counter()
    error = update.double() - xs.double().mean(0)
    nmse = (error.square().sum() / xs.double().square().sum(1).mean().clamp_min(1e-300)).item()
    logs = []
    for slot, (x, a, q, rec, p) in enumerate(zip(xs, aa, qs, recovered, payloads)):
        record = dict(client_slot=slot, metadata=p.metadata.tolist(), wire_bits=p.wire_bits,
                      diagnostics=p.diagnostics)
        if stage:
            scale = rms32(a)
            z = a / scale if scale != 0 else torch.zeros_like(a)
            record.update(rms=scale.item(), z_quantiles=torch.quantile(z, torch.tensor([.01, .5, .99])).tolist(),
                          coefficient_mse=(q.double()-a.double()).square().mean().item(),
                          normalized_coefficient_mse=((q.double()-a.double()).square().sum()/a.double().square().sum().clamp_min(1e-300)).item(),
                          client_nmse=((rec.double()-x.double()).square().sum()/x.double().square().sum().clamp_min(1e-300)).item())
        logs.append(record)
    timings['diagnostic_seconds'] = time.perf_counter() - tick
    return update, nmse, logs, timings


def verify_binding(root):
    root = Path(root)
    manifest = json.loads((root/'source_manifest.json').read_text(encoding='utf8'))
    for rel, digest in manifest['files'].items():
        if old.sha(Path(rel)) != digest:
            raise RuntimeError(f'Frozen source changed: {rel}')
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf8'))
    book = root/'assets/codebooks.json'
    if old.sha(book) != protocol['codebook_file_sha256']:
        raise RuntimeError('Frozen codebook bytes changed')
    if CodebookRegistry.load(book).fingerprint() != FINGERPRINT:
        raise RuntimeError('Frozen codebook fingerprint mismatch')
    return dict(head=manifest['head'], source_sha256=manifest['snapshot_sha256'],
                protocol_sha256=old.sha(root/'protocol.json'), codebook_fingerprint=FINGERPRINT)


def train_seed(root, seed, rounds=50, smoke=False):
    root = Path(root)
    binding = verify_binding(root)
    out = root/('smoke' if smoke else f'seed_{seed}')
    out.mkdir(exist_ok=False)
    (out/'stage_inputs').mkdir()
    registry = CodebookRegistry.load(root/'assets/codebooks.json')
    metadata = dict(status='running', seed=seed, rounds=rounds, smoke=smoke,
                    binding=binding, environment=old.environment(), clock='perf_counter serial CPU')
    old.write_json(out/'metadata.json', metadata)
    process_start = time.perf_counter()
    try:
        tick = time.perf_counter()
        x, y, tx, ty = old.load_data()
        torch.manual_seed(seed)
        base = MNIST_MLP()
        d = sum(p.numel() for p in base.parameters())
        assert d == 50890
        frame, signs = old.transforms(seed, d)
        models = {spec: copy.deepcopy(base) for spec in SPECS}
        budgets = dict.fromkeys(SPECS, 0)
        metadata.update(initialization_seconds=time.perf_counter()-tick,
                        initialization_sha256=tensor_sha(flatten_state_dict(base.state_dict())[0]))
        rng = np.random.default_rng(seed+8700)
        rows, logs, snapshots, round_io, diverged = [], [], [], [], {}
        for rnd in range(1, rounds+1):
            clients = rng.choice(10, 3, replace=False).tolist()
            batches = [old.minibatches(seed, rnd, c) for c in clients]
            for spec in SPECS:
                rep, bits, method = spec
                row = dict(seed=seed, round=rnd, canonical_config=key(spec), representation=rep,
                           bits=bits, method=method, clients=clients, status='complete',
                           aliases=[m for m in METHODS if m != method and canonical(m, bits) == method] if bits else [])
                if spec in diverged:
                    row.update(status='diverged', error=diverged[spec], accuracy=None, test_loss=None,
                               total_uplink_bits=budgets[spec])
                    rows.append(row)
                    continue
                model = models[spec]
                try:
                    tick = time.perf_counter()
                    xs = torch.stack([old.train_delta(model, x, y, batch) for batch in batches])
                    row['local_training_seconds'] = time.perf_counter()-tick
                    if not torch.isfinite(xs).all():
                        raise FloatingPointError('Non-finite local update')
                    row['snapshot_io_seconds'] = 0.
                    if rep == 'Original' and rnd in STAGES:
                        tick = time.perf_counter()
                        before = flatten_state_dict(model.state_dict())[0]
                        name = f'stage_inputs/round_{rnd}.pt'
                        torch.save(dict(updates=xs, global_before=before), out/name)
                        snapshots.append(dict(seed=seed, round=rnd, global_model_round=rnd-1,
                                              clients=clients, file=name, file_sha256=old.sha(out/name),
                                              tensor_sha256=tensor_sha(xs), global_sha256=tensor_sha(before), binding=binding))
                        row['snapshot_io_seconds'] = time.perf_counter()-tick
                    if rep == 'Original':
                        tick = time.perf_counter()
                        update = xs.mean(0)
                        row.update(aggregation_seconds=time.perf_counter()-tick, compression_seconds=0.,
                                   transform_seconds=0., encode_seconds=0., decode_seconds=0., inverse_seconds=0., diagnostic_seconds=0.)
                        nmse, diagnostics, used = 0., [], wire_bits(spec)
                    else:
                        update, nmse, diagnostics, times = instrumented_aggregate(
                            xs, rep, bits, method, frame, signs, registry, seed, rnd,
                            stage=(rnd in STAGES and method.startswith('fixed-')))
                        row.update(times)
                        used = sum(z['wire_bits'] for z in diagnostics)
                    if not torch.isfinite(update).all():
                        raise FloatingPointError('Non-finite reconstructed update')
                    tick = time.perf_counter()
                    old.apply_update(model, update)
                    row['server_apply_seconds'] = time.perf_counter()-tick
                    tick = time.perf_counter()
                    acc, loss = old.evaluate(model, tx, ty)
                    row['evaluation_seconds'] = time.perf_counter()-tick
                    if not np.isfinite([acc, loss, nmse]).all():
                        raise FloatingPointError('Non-finite test metric or distortion')
                    budgets[spec] += used
                    row.update(accuracy=acc, test_loss=loss, aggregate_nmse=nmse,
                               round_uplink_bits=used, total_uplink_bits=budgets[spec],
                               serial_algorithm_seconds=row['local_training_seconds']+row['compression_seconds']+
                               row['server_apply_seconds']+(row['aggregation_seconds'] if rep == 'Original' else 0.))
                    logs.extend(dict(seed=seed, round=rnd, canonical_config=key(spec), **z) for z in diagnostics)
                except FloatingPointError as error:
                    diverged[spec] = f'{type(error).__name__}: {error}'
                    row.update(status='diverged', error=diverged[spec], accuracy=None, test_loss=None,
                               total_uplink_bits=budgets[spec])
                rows.append(row)
            if registry.fingerprint() != FINGERPRINT:
                raise RuntimeError('In-memory frozen codebook changed')
            verify_binding(root)
            tick = time.perf_counter()
            old.write_json(out/'rounds.json', rows)
            old.write_json(out/'payloads.json', logs)
            old.write_json(out/'stage_inputs/manifest.json', snapshots)
            round_io.append(dict(round=rnd, shared_round_log_io_seconds=time.perf_counter()-tick))
            old.write_json(out/'io.json', round_io)
            print(f'seed={seed} round={rnd}/{rounds} paths={len(SPECS)} rows={len(rows)} elapsed={time.perf_counter()-process_start:.1f}s', flush=True)
        metadata.update(status='complete', process_wall_seconds=time.perf_counter()-process_start,
                        rows=len(rows), diverged_configs=[key(s) for s in diverged], final_binding=verify_binding(root))
        old.write_json(out/'metadata.json', metadata)
    except Exception as error:
        metadata.update(status='execution_failed', error=f'{type(error).__name__}: {error}',
                        process_wall_seconds=time.perf_counter()-process_start)
        old.write_json(out/'metadata.json', metadata)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.seed not in range(5):
        p.error('Formal seeds must be 0..4')
    torch.set_num_threads(2)
    train_seed(args.root, args.seed, 2 if args.smoke else 50, args.smoke)


if __name__ == '__main__':
    main()
