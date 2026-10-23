# Federated Learning Compression Experiments

This repository provides a single, modular workflow for federated MNIST and
EMNIST experiments. Compression is configured with one encoder, federated
training with one round runner, and experiment output with one CSV schema.
DRIVE and DRIVE+ are reserved encoder names; their algorithms are intentionally
deferred in this refactor.

## Quick start

Run a small IID MNIST experiment with raw 1-bit updates:

```powershell
python run_experiment.py --task mnist_mlp --transform raw --bits 1 `
  --num_clients 10 --clients_per_round 3 --total_rounds 3 `
  --images_per_client 120 --output_dir results/tutorial
```

The first run downloads MNIST through torchvision if it is not already in
`data/`. The run creates `hparams.json` and `metrics.csv` inside a directory
named after the experiment. `test_acc` is a fraction in `[0, 1]`; all BPD values
are per model dimension for one client update in one round.

To run with the MLP, CNN, or EMNIST CNN, use `--task mnist_mlp`,
`--task mnist_cnn`, or `--task emnist_cnn`. EMNIST accepts
`--emnist_split balanced` (47 classes) or `--emnist_split byclass` (62 classes).

## Select a compression method

```powershell
# No compression: 32 transmitted bits per parameter
python run_experiment.py --transform uncompressed --total_rounds 10

# Random-sign / SignSGD without a transform
python run_experiment.py --transform raw --scale_mode doscale --bits 1

# Randomized Hadamard transform followed by quantization
python run_experiment.py --transform hadamard --scale_mode doscale --bits 2

# Fourier-Kashin, with redundancy lambda = D/d
python run_experiment.py --transform fourier_kashin --bits 2 --lam 1.3

# Balanced Kashin solver
python run_experiment.py --transform fourier_kashin_balanced --bits 2 `
  --lam 1.3 --kashin_iters 10 --kashin_eta 0.9

# Fixed clipping range; no per-vector scale metadata
python run_experiment.py --transform hadamard --scale_mode noscale `
  --bits 1 --clip_val 0.25 --noscale_bound_mode dimension_scaled
```

`doscale` estimates a per-vector scale from the encoded coefficients. Select it
with `--scale_type l1_mean`, `l2_norm`, or `max` for 1-bit reconstruction
levels. Multibit `doscale` uses the full coefficient range `[-max(abs(x)),
max(abs(x))]` to avoid clipping bias. `noscale` sends no scale metadata and
defaults to `--noscale_bound_mode dimension_scaled`, where `--clip_val` is the
reference update L2 norm and the per-coordinate clipping range is adjusted for
the selected transform. `raw_fixed` uses `clip_val` directly for raw coordinates.
For `raw`, the transform is the identity; `hadamard` pads to the next power of
two; Fourier-Kashin uses `D=max(d, round(lam*d))`. BPD is `bits * actual_lam`
for noscale, `bits * actual_lam + 32/d` for doscale scale metadata, and 32 for
uncompressed.
The CLI accepts `drive` and `drive_plus` names to reserve the intended method
names, but currently raises a clear `NotImplementedError` for them.

## Reproducible federated setup

The CLI options cover the dataset partition, number of clients, selected
clients per round, local training, server learning rate, device and random seed.
For non-IID partitions, add `--partition dirichlet --dirichlet_alpha 0.5`.
Client sampling is without replacement within each round. With the default
`--frame_seed_mode independent`, compression seeds are derived per round and
client; use `shared` to reuse the round seed across clients.

The central entry point is:

```powershell
python run_experiment.py --help
```

## Run experiment grids and plot results

On Bash, run one declarative grid stage:

```bash
bash experiments.sh stage1_doscale
bash experiments.sh stage1_noscale
bash experiments.sh stage2_bpd_law
bash experiments.sh stage2_bpd_law_doscale
```

Set `TOTAL_ROUNDS`, `CLIENTS`, `PER_ROUND`, and `OUT` to adjust the grid. The
stages compare uncompressed, raw, Hadamard and Fourier-Kashin methods; the
DRIVE entries are omitted until implementation is added.

Generate summary figures after collecting runs:

```powershell
python plot_results.py --output_dir results/drive_style
```

This writes `accuracy_vs_rounds.png` (accuracy and NMSE against rounds and
cumulative BPD), `accuracy_vs_bpd.png` (last-ten-round mean accuracy against
effective BPD, including a Kashin Pareto frontier and the FedAvg baseline), and
`nmse_vs_bpd.png` (last-ten-round mean client NMSE against BPD).

## Project layout

- `compression/`: Fourier-Kashin frames and solvers, quantizers, Hadamard
  transforms, and the `VectorEncoder` registry.
- `federated/`: local client training, evaluation, and decoded-update averaging.
- `data/`: MNIST/EMNIST loading and IID/Dirichlet client partitions.
- `models/`: MLP and CNN classifiers.
- `run_experiment.py`: unified training CLI.
- `experiments.sh`: declarative Bash experiment stages.
- `plot_results.py`: plots all matching run folders under the selected output.
- `archive/`: previous one-off experiments and APIs, retained for reference
  and regression coverage. Use `run_experiment.py` for new runs.
- `experiments/diagnostics/`: retained frame benchmark diagnostic.

## Verify

```powershell
python -m pytest
python run_experiment.py --task mnist_mlp --transform raw --total_rounds 3 `
  --num_clients 3 --clients_per_round 2 --images_per_client 64 `
  --batch_size 32 --output_dir results/smoke
python plot_results.py --output_dir results/smoke
```
