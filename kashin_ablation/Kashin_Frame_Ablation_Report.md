# Kashin Frame Ablation Report

## Objective

Compare alternative Kashin frames independently before changing the FL pipeline.

## Methods

Fourier uses random phases and FFT, supports arbitrary D, and avoids dense matrices. Gaussian QR constructs a dense Gaussian matrix and is limited to small validation dimensions.

## Results

See `results/frame_comparison.md` and `results/frame_comparison.csv`.

## Recommendation

Fourier is recommended for the subsequent FL experiment because it supports arbitrary D and has FFT-scale computational cost. Gaussian QR is useful as a small-scale reference, but its dense construction is not practical for the real model dimension.
