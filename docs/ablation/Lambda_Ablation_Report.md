# Fourier-Kashin Lambda Ablation

## 1. Objective

Study how `lambda = D / d` affects coefficient flattening, low-bit quantization, and communication. No lambda is assumed optimal in advance.

## 2. Experimental Setup

- d = 50890
- lambda = 1.25, 1.5, 2.0, 2.5, 3.0, 4.0
- bits = 1, 2, 4, 8
- input = RealFL
- iterations = 10
- seed = 2026

## 3. Lambda vs D

| lambda | D | D/d |
|---:|---:|---:|
| 1.25 | 63613 | 1.2500 |
| 1.5 | 76335 | 1.5000 |
| 2.0 | 101780 | 2.0000 |
| 2.5 | 127225 | 2.5000 |
| 3.0 | 152670 | 3.0000 |
| 4.0 | 203560 | 4.0000 |

## 4. Coefficient Flattening

| lambda | initial peak | Kashin peak | peak ratio |
|---:|---:|---:|---:|
| 1.25 | 0.0209101 | 0.0151588 | 0.724950 |
| 1.5 | 0.02118 | 0.0119644 | 0.564891 |
| 2.0 | 0.0230582 | 0.00910503 | 0.394871 |
| 2.5 | 0.0199461 | 0.00671281 | 0.336547 |
| 3.0 | 0.0235542 | 0.00619882 | 0.263172 |
| 4.0 | 0.0224171 | 0.00488619 | 0.217967 |

## 5. Quantization

| lambda | 1-bit error | 2-bit error | 4-bit error | 8-bit error |
|---:|---:|---:|---:|---:|
| 1.25 | 2.94074 | 0.687857 | 0.134119 | 0.00786643 |
| 1.5 | 2.78388 | 0.621996 | 0.120216 | 0.00711376 |
| 2.0 | 2.43102 | 0.509065 | 0.0996633 | 0.00587814 |
| 2.5 | 2.30217 | 0.445331 | 0.0876143 | 0.00512808 |
| 3.0 | 2.46025 | 0.450059 | 0.0861194 | 0.00505415 |
| 4.0 | 2.61883 | 0.435543 | 0.0795321 | 0.00467211 |

## 6. Communication

For each configuration, coefficient payload is `D * bits`; no additional metadata is counted because this ablation uses the shared fixed construction and the existing scalar quantizer protocol.

## 7. Computation Cost

See `lambda_vs_total_time.png` and the CSV for initialization, analysis, solver, synthesis, and total times.

## 8. Trade-off Analysis

Increasing lambda increases D and communication. The measured peak ratio and reconstruction error determine whether the additional coefficients are worthwhile. Comparisons across lambda are made separately for each bitwidth.

## 9. Pareto-efficient Configurations

| lambda | bits | communication | relative error |
|---:|---:|---:|---:|
| 1.25 | 1 | 63613 | 2.94074 |
| 1.5 | 1 | 76335 | 2.78388 |
| 2.0 | 1 | 101780 | 2.43102 |
| 2.5 | 1 | 127225 | 2.30217 |
| 1.25 | 2 | 127226 | 0.687857 |
| 1.5 | 2 | 152670 | 0.621996 |
| 2.0 | 2 | 203560 | 0.509065 |
| 2.5 | 2 | 254450 | 0.445331 |
| 1.25 | 4 | 254452 | 0.134119 |
| 1.5 | 4 | 305340 | 0.120216 |
| 2.0 | 4 | 407120 | 0.0996633 |
| 2.5 | 4 | 508900 | 0.0876143 |
| 1.25 | 8 | 508904 | 0.00786643 |
| 1.5 | 8 | 610680 | 0.00711376 |
| 2.0 | 8 | 814240 | 0.00587814 |
| 2.5 | 8 | 1017800 | 0.00512808 |
| 3.0 | 8 | 1221360 | 0.00505415 |
| 4.0 | 8 | 1628480 | 0.00467211 |

## 10. Recommendation

The next FL experiment should test the lambda values that are Pareto-efficient at the target communication budget. This report does not preselect lambda=2; the recommendation must follow the generated table and plots.
