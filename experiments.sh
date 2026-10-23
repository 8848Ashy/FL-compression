#!/usr/bin/env bash
# Declarative experiment stages for the unified runner.
# Set TOTAL_ROUNDS / CLIENTS / PER_ROUND to shorten a run during development.
set -euo pipefail

TOTAL_ROUNDS=${TOTAL_ROUNDS:-100}
CLIENTS=${CLIENTS:-100}
PER_ROUND=${PER_ROUND:-10}
OUT=${OUT:-./results/drive_style}
COMMON=(--task mnist_mlp --partition iid --num_clients "$CLIENTS"
        --clients_per_round "$PER_ROUND" --total_rounds "$TOTAL_ROUNDS"
        --output_dir "$OUT")

stage1_doscale() {
  for transform in uncompressed raw hadamard fourier_kashin; do
    python run_experiment.py "${COMMON[@]}" --transform "$transform" \
      --scale_mode doscale --bits 1 --lam 1.3 --exp_name "stage1_doscale_${transform}"
  done
  # DRIVE and DRIVE+ are intentionally omitted until their algorithms are ported.
}

stage1_noscale() {
  for transform in raw hadamard fourier_kashin; do
    python run_experiment.py "${COMMON[@]}" --transform "$transform" \
      --scale_mode noscale --clip_val 0.02 --bits 1 --lam 1.3 \
      --exp_name "stage1_noscale_${transform}"
  done
}

stage2_bpd_law() {
  for bits in 1 2 3; do
    for transform in raw hadamard; do
      python run_experiment.py "${COMMON[@]}" --transform "$transform" \
        --scale_mode noscale --clip_val 0.02 --bits "$bits" --lam 1.3 \
        --exp_name "stage2_${transform}_b${bits}"
    done
    for lam in 1.15 1.3 1.6 2.0; do
      python run_experiment.py "${COMMON[@]}" --transform fourier_kashin \
        --scale_mode noscale --clip_val 0.02 --bits "$bits" --lam "$lam" \
        --exp_name "stage2_fourier_kashin_b${bits}_lam${lam}"
    done
  done
}

case "${1:-}" in
  stage1_doscale) stage1_doscale ;;
  stage1_noscale) stage1_noscale ;;
  stage2_bpd_law) stage2_bpd_law ;;
  *) echo "Usage: $0 {stage1_doscale|stage1_noscale|stage2_bpd_law}" >&2; exit 2 ;;
esac
