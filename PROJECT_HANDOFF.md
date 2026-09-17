# FL Kashin Compression Project — Current Handoff

**Updated:** 2026-09-17
**Project root:** `D:\FL` (dev), `/home/hczhang/FL` (GPU server, env `/home/hczhang/flenv`)  
**Python environment:** `C:\Users\zhang\.conda\envs\fl_env\python.exe`

## 1. Research objective

Study uplink communication compression for federated learning on MNIST. The intended scientific comparison is:

1. Original FedAvg (uncompressed model updates)
2. SRK (Hadamard/FWHT rotation + stochastic k-level quantization)
3. Fourier-Kashin (redundant Fourier frame + Kashin coefficient solver + stochastic k-level quantization)

The research question is **not** “does Kashin always beat SRK at the same bit width?” Kashin uses a redundant representation, so the meaningful question is whether it can use fewer quantization bits while retaining comparable accuracy at lower actual communication.

Do not claim Kashin is better unless the observed accuracy difference and communication budget support it.

## 2. Development, GitHub, and server deployment workflow

**`D:\FL` is the only place to author code.** Do not edit source, configuration, or experiment scripts directly on the GPU server or through its mounted drive. Every code change follows one route:

```text
D:\FL edit → local quick checks → commit → push GitHub → server pull → verify same commit → run → collect results
```

Before local work, run `git status --short` and `git pull --ff-only origin master`. Before a server run, record `git rev-parse --short HEAD`; it must equal the commit just pushed from `D:\FL`. Do not use a merge pull, reset, force-push, or manual file copying to resolve a version mismatch.

If an emergency server edit is unavoidable, stop before running an experiment: copy the change back to `D:\FL`, review it, commit and push locally, then re-pull the committed version on the server. An uncommitted server edit is never a valid experiment source.

Generated results are ignored by Git. For every meaningful server run, keep a timestamped output directory and add a concise `OPENCODE_SESSION_LOG.md` entry containing the commit, command, environment/GPU, key configuration, output path, and conclusion. Copy selected results back locally for review without overwriting history.

### Standard commands

Local (`D:\FL`):

```powershell
git pull --ff-only origin master
& "C:\Users\zhang\.conda\envs\fl_env\python.exe" tests/test_refactor_regression.py
& "C:\Users\zhang\.conda\envs\fl_env\python.exe" tests/test_crn_relerr.py
git add <intended files>
git commit -m "<clear change summary>"
git push origin master
git rev-parse --short HEAD
```

Server (`~/FL`):

```bash
git pull --ff-only origin master
git rev-parse --short HEAD
# Confirm this equals the local pushed hash before running.
```

## 3. Fixed experimental setup

- Dataset: MNIST
- Model: MLP `784 -> 64 -> 10`
- Parameter dimension: `d = 50890`
- Clients: 10
- Data per client: 600 consecutive MNIST training examples
- Test set: first 10000 MNIST test examples (full test set; raised from 1000 to cut evaluation noise to ±0.27%)
- Local training: SGD, learning rate 0.05, 2 local epochs
- Default main rounds: 50 (`NUM_ROUNDS_FOCUS`)
- Seeds: model seed 42; Fourier-Kashin experiment seed 2026

Updates, not full model parameters, are uploaded:

```text
delta_w_i = local_model_i - global_model
global_next = global + average(compressed(delta_w_i))
```

Without compression, averaging deltas then adding back is numerically equivalent to averaging local models (tested at approximately `1e-9` maximum difference).

## 4. Current project architecture

```text
2017.py                    thin main entry
config.py                  experiment switches and constants
models/mnist_mlp.py        MNIST_MLP
data/mnist_federated.py    MNIST loading and fixed client split
federated/                 local training, evaluation, Original aggregation
compression/               SRK, Fourier-Kashin, quantizers, solver
utils/                     state_dict and communication formulas
experiments/               experiment scripts
tests/                     unit/regression checks
kashin_ablation/           standalone Fourier lambda/frame ablation
results/, plots/           generated main-experiment outputs
```

### Important implementation distinction

- **SRK** remains the Hadamard/FWHT baseline. It zero-pads `d=50890` to `65536` and uses FWHT.
- **Fourier-Kashin** is now `compression.kashin_frame.FourierKashinFrame`. It always uses FFT/IRFFT, even when `D` is a power of two. It has no FWHT fallback.
- `D` can be any integer: `D = round(lambda * d)`.
- Main selected lambda is configured in `config.py` as `KASHIN_LAMBDA = 2.0`.

## 5. Current main runnable experiment

Run from `D:\FL`:

```powershell
$env:MPLBACKEND="Agg"
& "C:\Users\zhang\.conda\envs\fl_env\python.exe" 2017.py
```

Current `config.py` has `RUN_FULL_EXPERIMENT = True`, `RUN_LOWBIT_EXPERIMENT = True`, `NUM_ROUNDS_FOCUS = 50`, `CRN_PAIRED = True`, and `LOWBIT_NUM_CLIENTS = 3`. The entry runs a **50-round paired SRK/Fourier-Kashin lambda sweep** as a three-client sensitivity experiment, plus the uncompressed Original reference. The fixed 10-client setup remains the baseline configuration in `NUM_CLIENTS` and is not silently replaced.

```text
Original
SRK at 2, 4, 8, and 16 bits
Kashin at 2, 4, and 8 bits, each with lambda = 1, 1.5, 2, 2.5, and 3
```

With `CRN_PAIRED=True`, every variant receives the same per-round client shuffle and quantizer random draws. This improves comparison precision; it does not change the algorithms. The run writes timestamp-prefixed figures and these tabular summaries:

- `results/kashin_lowbit_round_metrics.csv`
- `results/kashin_accuracy_saving.csv`
- `results/kashin_relerr2_summary.csv`
- `results/fixed_budget_accuracy.csv`
- `results/fixed_budget_frontier.csv`

The old cumulative-communication/round-accuracy and target-accuracy figures are no longer generated; fixed-budget accuracy is the primary task-level tradeoff figure. The tracked local figures are historical pre-CRN outputs, not results of the current code. Preserve every new server run under a timestamped directory and record its commit.

## 6. Communication accounting (authoritative)

All formulas are centralized in `utils/communication.py`.

```text
Original per client per round = d * 32
SRK per client per round      = 65536 * bits + 64
Kashin per client per round   = D * bits + 64
D                              = round(lambda * d)

total bits = per-client-per-round bits * clients * rounds
MB         = total bits / 8 / 1024 / 1024
```

For this model:

| Method | Encoded dimension | 1-bit | 2-bit | 4-bit |
|---|---:|---:|---:|---:|
| SRK | 65536 | 65600 | 131136 | 262208 |
| Kashin, lambda=2 | 101780 | 101844 | 203624 | 407184 |
| Kashin, lambda=2.5 | 127225 | 127289 | 254514 | 509064 |
| Kashin, lambda=3 | 152670 | 152734 | 305404 | 610744 |

These entries are **bits per client per round**. Never claim SRK and Kashin have the same communication merely because they use the same quantizer bit width.

## 7. Current quantitative findings

### Historical low-bit experiment (single seed, 8 rounds; pre-CRN)

Final values in `results/kashin_bit_tradeoff.csv`:

| Comparison | SRK accuracy | Kashin accuracy | SRK total MB | Kashin total MB | Saving |
|---|---:|---:|---:|---:|---:|
| SRK-2bit vs Kashin-1bit | 89.5% | 8.5% | 1.2506 | 0.9713 | 22.34% |
| SRK-4bit vs Kashin-2bit | 89.2% | 89.1% | 2.5006 | 1.9419 | 22.34% |
| SRK-8bit vs Kashin-4bit | 89.3% | 89.2% | 5.0006 | 3.8832 | 22.35% |

Defensible interpretation:

- Kashin at 1 bit fails in the current min/max stochastic quantizer pipeline (accuracy collapsed to ~8.5%).
- Kashin 2-bit is close to SRK 4-bit in this single run while sending ~22.3% less traffic.
- Kashin 4-bit is close to SRK 8-bit in this single run while sending ~22.3% less traffic.
- This is single-seed MNIST/MLP evidence only. It is not enough for a general superiority claim.

### Variance reduction and distortion instrumentation (2026-09-15)

- Multi-seed (5 seeds, old 8-round/1000-test config) showed every config within 89.4–89.9% with std 0.19–0.51%: at b>=2 the accuracy differences are below the noise floor. Diagnosis: per-client relative squared compression error at b=2 is already ~1% (`results/quantizer_distortion.csv`), and 10-client averaging shrinks it further, so test accuracy cannot resolve bits>=2 or lambda effects.
- New optional instrumentation on `federated_round_srk_update` / `federated_round_kashin_update`: `quant_seeds` (per-client seeds feeding a dedicated quantizer generator) and `relerr_out` (list that receives per-client `||delta_hat - delta||^2 / ||delta||^2`). Default call signature behavior is unchanged.
- New common-random-number (CRN) pairing, toggled by `config.CRN_PAIRED = True`: `build_mnist_federated_data(..., paired_shuffle=True)` gives each client loader its own generator; `experiments/lowbit.py` re-seeds shuffle per (round, client) and quantization per (round, client), so all compared variants see identical data order and quantizer randomness. Each run is still stochastic; only the comparison is paired (variance reduction, not determinism).
- New outputs: `relerr2` / `relerr2_max` columns in `kashin_lowbit_round_metrics.csv`, `results/kashin_relerr2_summary.csv`, `plots/kashin_relerr2_tradeoff_*.png`, `plots/multi_seed_relerr2.png`, relerr columns in `multi_seed_*.csv`.
- Use the relerr2 (distortion) curves as the primary evidence for lambda/bits trends; accuracy remains endpoint validation.

### Fourier lambda ablation (frame-level, not full FL)

`kashin_ablation/frame_benchmark.py` runs a real-update Fourier frame diagnostic across lambda values. The generated lambda report and plots are under `kashin_ablation/results/`.

Observed trend: larger lambda reduces the coefficient peak ratio but increases coefficient count and communication. It does **not** establish the best lambda for full FL by itself.

## 8. Current quantization status

The active SRK/Kashin main experiment uses:

```python
stochastic_k_level_quantize
```

Location: `compression/quantization.py`.

Lloyd-Max helpers exist in `compression/quantization.py` and `compression/lloyd_max.py`, but the shared-codebook LM FL experiment is currently **not connected**. `experiments/lloyd_max.py` is an unfinished placeholder. Do not state that LM is active in the current main experiment.

## 9. Verified module status and known gaps

### Connected and usable

- `models/mnist_mlp.py`
- `data/mnist_federated.py`
- `federated/local_training.py`
- `federated/evaluation.py`
- `federated/aggregation.py` (Original update aggregation)
- `compression/srk.py`
- `compression/kashin_frame.py`
- `compression/kashin_solver.py`
- `compression/kashin_compressor.py`
- `utils/state_dict.py`
- `utils/communication.py`
- `experiments/lowbit.py`
- `tests/test_refactor_regression.py`
- `tests/test_crn_relerr.py`

### Present but not fully connected / placeholders

- `experiments/communication_tradeoff.py` — placeholder; it does not run the Original/SRK/Kashin unified trade-off yet.
- `experiments/coefficient_distribution.py` — placeholder.
- `experiments/lloyd_max.py` — placeholder; shared-codebook calibration is not restored.
- `experiments/legacy_icml2017.py` and `compression/legacy.py` — historical placeholders; old SK/SVK pipeline not restored.
- `utils/debug_metrics.py` — unused.

Consequently, the current source can run the **low-bit SRK/Kashin experiment**, but does not yet provide a restored unified `Original vs SRK vs Fourier-Kashin` main experiment through `experiments/communication_tradeoff.py`.

## 10. Tests and safe commands

Compile and run regression tests (no full FL):

```powershell
& "C:\Users\zhang\.conda\envs\fl_env\python.exe" -m compileall -q config.py models data compression federated utils experiments tests 2017.py
& "C:\Users\zhang\.conda\envs\fl_env\python.exe" tests/test_refactor_regression.py
& "C:\Users\zhang\.conda\envs\fl_env\python.exe" tests/test_crn_relerr.py
```

Regression tests verify state dict handling, stochastic quantization validity, Fourier frame reconstruction for arbitrary D, Original/SRK/Kashin one-round aggregation, communication formulas, and FFT-vs-FWHT separation.

The environment does not currently have pytest installed; run test files directly or use the regression script.

## 11. Suggested next work (do not assume authorization)

Recommended order:

1. Restore `experiments/communication_tradeoff.py` with a proper `Original / SRK / Fourier-Kashin` experiment using centralized communication accounting.
2. Add loss reporting if required; current evaluation returns accuracy only.
3. Restore the LM shared-codebook experiment in `experiments/lloyd_max.py`, including explicit codebook communication accounting and privacy discussion.
4. Run multiple seeds only after the main comparison is stable.
5. Consider a harder dataset/model only after current MNIST results and communication accounting are reproducible.

## 12. Rules for the next AI

- Do not change model, MNIST partition, optimizer, local epochs, SRK transform, stochastic quantizer, or Kashin solver merely to obtain favorable results.
- Do not add Gaussian/GaussianQR or caches.
- Do not use FWHT/Hadamard in `FourierKashinFrame`; FWHT belongs only to SRK.
- Do not run long experiments unless explicitly asked.
- Do not overwrite historical result files without warning.
- Do not claim Kashin superiority from one random seed or from unequal accuracy targets.
- CRN pairing and relerr instrumentation are measurement tools only; do not let them alter model, optimizer, SRK transform, quantizer, or Kashin solver mathematics.
- Use `KASHIN_LAMBDA` from `config.py`, not hard-coded `D=65536`, for new Fourier-Kashin experiments.
- Before modifying code, synchronize `D:\FL` with GitHub using `git pull --ff-only origin master`; do not write code on the server as a shortcut.
- Before every server experiment, verify the server HEAD equals the local pushed commit and record that hash with the results.
- `PROJECT_HANDOFF.md` is a maintained project specification: update it only when a committed change makes its workflow, configuration, architecture, or conclusions inaccurate. Append ordinary run notes to `OPENCODE_SESSION_LOG.md` instead.

## 13. Copy/paste prompt for the next chat

```text
I am continuing a modular federated-learning Kashin compression project in D:\\FL.
First run `git status --short` and `git pull --ff-only origin master`, then read D:\\FL\\PROJECT_HANDOFF.md and treat it as the current source of truth.
Do not modify code yet. Inspect the requested module(s), identify the smallest safe next step in Chinese, and do not run a long experiment unless I explicitly ask.
For every code change, use only D:\\FL → quick checks → commit → push → server `git pull --ff-only` → verify matching commit → run. Do not edit server code directly. Record every meaningful server result with its Git hash in OPENCODE_SESSION_LOG.md.
Important: SRK uses Hadamard/FWHT; Fourier-Kashin must always use FFT for arbitrary D. The active experiment uses stochastic k-level quantization, CRN pairing, and per-round relerr2 instrumentation—not Lloyd-Max.
```
