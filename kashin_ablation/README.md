# Kashin Frame Ablation Study

This directory independently compares two Kashin frames:

- Random-phase Fourier tight frame, supporting arbitrary `D` without FWHT;
- Gaussian QR tight frame, intended only for small-scale validation.

The experiment is independent from `2017.py` and does not modify the main FL pipeline.

Run from `D:\FL`:

```text
python kashin_ablation/frame_benchmark.py
```

Outputs are written to `kashin_ablation/results/`.

