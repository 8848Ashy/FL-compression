# Kashin Frame Ablation Report

## Objective

Compare Random Fourier and Pure Gaussian Random frames independently before changing the FL pipeline. `2017.py` is not modified.

## Methods

Fourier uses random phases and FFT, supports arbitrary D, and avoids dense matrices. Pure Gaussian uses G_ij~N(0,1/D), regenerated in deterministic chunks; no QR and no dense D×d storage are used. Both are scaled with A=D/d.

## Experimental Setup

The requested real-model dimension is d=50890, with lambda=1.5, 2, 3 and D=int(lambda*d). Inputs are Gaussian, sparse, and a real FL delta_w.

## Results

The required large-scale run is recorded in `results/large_scale_fourier_vs_gaussian.csv` and `.md`. Fourier completed. Gaussian block initialization produced about 19.3 GB of cache, but the matrix products did not finish within the runtime window and were stopped. The Gaussian row is explicitly marked `TIME_LIMIT`; no quality numbers were fabricated.

## Runtime Limitation

Chunked caching avoids allocating the full `D x d` matrix in RAM, but it still requires about 19.3 GB disk for this setting. Analysis and synthesis must multiply all Gaussian blocks, and the Kashin solver performs ten residual iterations. This is a compute-time bottleneck, not an OOM result.

## Recommendation

No frame is selected automatically. Use the tables to decide after considering peak flattening, quantization error, runtime, and the recorded large-scale Gaussian cost or limitation.
