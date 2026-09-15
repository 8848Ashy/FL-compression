# Kashin Low-Bit Experiment

## Motivation

Same-bit SRK/Kashin is not the main comparison because Kashin transmits a redundant frame with `D > d`. This experiment tests the actual hypothesis: Kashin may maintain similar accuracy with fewer quantization bits.

## Communication formulas

SRK uses `65536 * bits + 64` bits per client per round. Kashin with lambda=2 uses `101780 * bits + 64` bits per client per round. All totals include 10 clients and 8 rounds.

## Final comparison

| Baseline | Kashin | Baseline accuracy | Kashin accuracy | Baseline MB | Kashin MB | Accuracy difference | Saving |
|---|---|---:|---:|---:|---:|---:|---:|
| SRK-2bit | Kashin-1bit | 89.5% | 8.5% | 1.2506 | 0.9713 | -81.0 percentage points | 22.34% |
| SRK-4bit | Kashin-2bit | 89.2% | 89.1% | 2.5006 | 1.9419 | -0.1 percentage points | 22.34% |
| SRK-8bit | Kashin-4bit | 89.3% | 89.2% | 5.0006 | 3.8832 | -0.1 percentage points | 22.35% |

## Conclusion

The 2-bit-to-1-bit comparison does not support a Kashin advantage: Kashin-1bit collapsed to 8.5% accuracy. The 4-bit-to-2-bit and 8-bit-to-4-bit comparisons satisfy the predefined criterion (accuracy loss no more than 0.5 percentage points and positive communication saving), with approximately 22.3% lower communication. These conclusions are for this single-seed MNIST experiment and should not be generalized without additional seeds.

Detailed round results are in `kashin_lowbit_round_metrics.csv`, target-accuracy results in `kashin_accuracy_saving.csv`, and pair comparisons in `kashin_bit_tradeoff.csv`.
