# Large-scale Fourier vs Pure Gaussian Frame

Settings: `d=50890`, `lambda=2`, `D=101780`, `input=RealFL`, `bits=2`, Kashin iterations=10, seed=2026.

| Frame | Init/cache | Analysis | Solver | Synthesis | Peak ratio | 2-bit MSE | Relative reconstruction | Peak memory | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Fourier | 0.038s | 0.036s | 0.059s | 0.002s | 0.3949 | 5.596e-6 | 2.919e-7 | 145.4 MB | COMPLETED |
| GaussianRandom | 19.3 GB block cache | not completed | not completed | not completed | - | - | - | - | TIME_LIMIT |

## Interpretation

The Fourier result completed successfully. The Gaussian frame was initialized using deterministic block files, so its random matrix was generated only once and reused; however, the subsequent large matrix products did not finish within the runtime window. This is a compute-time bottleneck, not an OOM result. No Gaussian quality numbers are reported.
