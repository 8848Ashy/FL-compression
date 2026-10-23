"""Local-only reproducible 5x50 driver; rejects existing output directories."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import platform
import importlib.metadata
from pathlib import Path

import torch

from archive.experiments import quantizer_study as old
from archive.experiments.full_quantizer_training import FINGERPRINT, SPECS, key, verify_binding
from archive.experiments.summarize_quantizer_training import PRIMARY, validate
from archive.compression.quantizer_payload import CodebookRegistry


def run(root, name, arguments):
    command = [sys.executable, *arguments]
    commands_path = root/'commands.json'
    commands = json.loads(commands_path.read_text(encoding='utf8')) if commands_path.exists() else []
    commands.append(dict(name=name, cwd=str(Path.cwd()), argv=command))
    old.write_json(commands_path, commands)
    with (root/'logs'/f'{name}.log').open('w', encoding='utf8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace')
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line.rstrip(), flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'{name} failed with exit {code}; see logs/{name}.log')


def freeze(root):
    files = sorted({p for folder in ('compression', 'experiments', 'federated', 'models', 'utils', 'tests')
                    for p in Path(folder).rglob('*.py')} | set(Path('.').glob('*.py')))
    entries = {}
    for path in files:
        rel = path.as_posix()
        destination = root/'source_snapshot'/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        entries[rel] = old.sha(path)
    def git(*args):
        return subprocess.check_output(['git', *args], text=True, encoding='utf8', errors='replace').strip()
    manifest = dict(head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                    status=git('status', '--short'), files=entries,
                    snapshot_sha256=hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest())
    (root/'tracked_diff.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary']))
    old.write_json(root/'source_manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(exist_ok=False, parents=True)
    (root/'assets').mkdir()
    (root/'logs').mkdir()
    book = Path('outputs/results/quantizer_study_20261004/calibration_v2/codebooks.json')
    if CodebookRegistry.load(book).fingerprint() != FINGERPRINT:
        raise RuntimeError('Old frozen codebook fingerprint mismatch; do not recalibrate')
    shutil.copy2(book, root/'assets/codebooks.json')
    for source in ('docs/Quantizer_Study_20261004_Report.md', 'docs/Quantizer_Study_20261004_Commands.md'):
        shutil.copy2(source, root/'assets'/Path(source).name)
    shutil.copy2('C:/Users/Torres/Downloads/Codex_FL_Full_Training_Task.md', root/'assets/Task.md')
    torch.set_num_threads(2)
    protocol = dict(version='2026-10-08', seeds=list(range(5)), rounds=50, unique_specs=[key(s) for s in SPECS],
        model='MLP 784-ReLU64-10', parameters=50890, D=65536, lambda_exact='65536/50890',
        data='MNIST first 6000 train, all 10000 test, contiguous client blocks600, (x/255-.1307)/.3081',
        dataset_hashes={str(f):old.sha(f) for f in sorted(Path('data/MNIST/raw').glob('*')) if f.is_file()},
        clients=10, clients_per_round=3, local_epochs=2, batch_size=32, optimizer='SGD', learning_rate=.05,
        device='cpu', torch_threads=2, solver='Fourier Kashin balanced 20 iterations, decay .9; existing range objective',
        fit='Existing six starts, maximum30 iterations per start; unchanged stopping rules',
        random_rules='torch.manual_seed(seed) init; clients numpy seed+8700; minibatches seed*100000+round*100+client; transforms seed+4000; rounding seed*100000+round*100+slot+9000000',
        primary_endpoint='round50 test accuracy', auxiliary=['mean accuracy rounds46..50', 'round50 test CE'],
        primary_comparisons=PRIMARY, thresholds=[.85, .9, .92], sustained='3 consecutive rounds; charged at third confirmation',
        statistics='new-reference pp; seed unit; sampleSD and unadjusted t(.975,n-1)CI; n<2 noCI; no equivalence/noninferiority',
        secondary='all B-A,C-B,D-C(2bit),E-A,F-A(2bit),Kashin-SRK,each-Original; no selection by result',
        target_failures=['not_reached', 'diverged', 'execution_failed'], stages=[1,15,30,50],
        communication='actual logical payload; no real network; fixed preinstalled20float32 centers640bits,10clients distribution>=6400bits separate',
        frozen_calibration='old seeds100/101 rounds1/2; never recalibrate', codebook_fingerprint=FINGERPRINT,
        codebook_file_sha256=old.sha(root/'assets/codebooks.json'), environment=old.environment(),
        dependency_versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','torchvision','matplotlib','pytest')},
        interpreter=sys.executable, cpu=platform.processor(),
        timing='exclusive transform/encode/decode/inverse/aggregation clocks; direct compression; separate apply/eval/diagnostics/snapshotIO/sharedIO; legacy internal temporary reconstructions stay in encode')
    old.write_json(root/'protocol.json', protocol)
    freeze(root)
    manifest = dict(status='preflight', binding=verify_binding(root), formal_runs=[])
    old.write_json(root/'run_manifest.json', manifest)
    try:
        run(root, 'tests', ['-m', 'pytest', 'tests', '-q'])
        run(root, 'smoke', ['-u', '-m', 'experiments.full_quantizer_training', '--root', str(root), '--seed', '0', '--smoke'])
        smoke = json.loads((root/'smoke/rounds.json').read_text(encoding='utf8'))
        smoke_integrity = validate(smoke, [0], 2)
        if smoke_integrity['failed_rows']:
            raise RuntimeError('Smoke has failed/diverged rows')
        old.write_json(root/'smoke/integrity.json', smoke_integrity)
        # Final snapshot after smoke. It must remain identical to the preflight version.
        before = manifest['binding']
        freeze(root)
        if verify_binding(root) != before:
            raise RuntimeError('Source/protocol changed during preflight')
        manifest['status'] = 'formal_running'
        for seed in range(5):
            entry = dict(seed=seed, directory=f'seed_{seed}', status='running')
            manifest['formal_runs'].append(entry)
            old.write_json(root/'run_manifest.json', manifest)
            run(root, f'seed_{seed}', ['-u', '-m', 'experiments.full_quantizer_training', '--root', str(root), '--seed', str(seed)])
            rows = json.loads((root/f'seed_{seed}/rounds.json').read_text(encoding='utf8'))
            entry.update(status='complete', integrity=validate(rows, [seed], 50), binding=verify_binding(root))
            old.write_json(root/'run_manifest.json', manifest)
        manifest['status'] = 'training_complete'
        old.write_json(root/'run_manifest.json', manifest)
        run(root, 'analysis', ['-u', '-m', 'experiments.summarize_quantizer_training', '--root', str(root)])
        manifest['status'] = 'complete'
        old.write_json(root/'run_manifest.json', manifest)
        print(f'COMPLETE {root}', flush=True)
    except Exception as error:
        manifest.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
        if manifest['formal_runs'] and manifest['formal_runs'][-1]['status'] == 'running':
            manifest['formal_runs'][-1]['status'] = 'execution_failed'
        old.write_json(root/'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
