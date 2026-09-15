# FL Communication Trade-off Report

## Experimental setup

The experiment compares Original FedAvg, SRK, and Fourier-Kashin on the existing MNIST FL pipeline. The dataset, model, clients, local epochs, optimizer, quantizer, and aggregation mathematics were not changed.

- Original parameter dimension: `d = 50890`
- Clients: 10
- Rounds: 8
- Kashin redundancy values: `lambda = 2, 2.5, 3`
- Quantization bits: `1, 2, 4`
- Kashin iterations: 10
- Random seed: 2026

## Communication accounting

All traffic is counted as absolute transmitted bits from every client in every round.

| Method | Encoded dimension | Bits per client per round |
|---|---:|---:|
| Original | 50890 | `50890 × 32 = 1,628,480` |
| SRK | 65536 | `65536 × bits + 64` |
| Kashin, lambda=2 | 101780 | `101780 × bits + 64` |
| Kashin, lambda=2.5 | 127225 | `127225 × bits + 64` |
| Kashin, lambda=3 | 152670 | `152670 × bits + 64` |

The reported total communication is:

`per-client-per-round bits × 10 clients × elapsed rounds`.

Normalized communication is retained separately as cumulative bits per original dimension per client.

## Results

Detailed per-round values are in `results/fl_round_metrics.csv`. The duplicate-format trade-off file is `results/communication_tradeoff.csv`. Target accuracy lookup is in `results/target_accuracy_communication.csv`.

The plots are:

- `plots/accuracy_vs_total_communication.png`
- `plots/accuracy_vs_normalized_communication.png`
- `plots/communication_to_target_accuracy.png`

The target file records `NOT_REACHED` when a method does not reach 90% or 92%; no interpolation was used.

## Interpretation

The total-communication plot is the primary practical comparison. The normalized plot is retained for theoretical comparison. SRK padding is explicitly counted, and Kashin communication increases with lambda because its transmitted coefficient dimension `D` increases. Therefore SRK and Kashin are not forced to have equal communication budgets.

No communication-saving claim is made unless the compared entries reach the same target accuracy. The next step is to choose target-accuracy operating points from the generated table and, if needed, repeat the selected lambda values with additional random seeds.
