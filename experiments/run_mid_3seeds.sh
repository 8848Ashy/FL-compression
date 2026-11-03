#!/bin/bash
# 3-seed 全量编排：Q1 -> 阶段汇总+b1误差棒图 -> Q2 -> 最终两个CSV
set -uo pipefail
cd ~/FL-compression
source ~/venvs/fl/bin/activate
ROOT=results/stage4_mid_full_lr10

echo "===== QUEUE 1 start $(date) ====="
python experiments/run_parallel.py --jobs experiments/jobs_mid3s_q1.json --workers 6 --device cpu --skip-existing

echo "===== INTERIM: all-seeds CSV + b1 errorbar $(date) ====="
python experiments/collect_table.py --root $ROOT --out $ROOT/master_table_lr10_all_seeds.csv
python experiments/plot_b1_errorbar.py --root $ROOT

echo "===== QUEUE 2 start $(date) ====="
python experiments/run_parallel.py --jobs experiments/jobs_mid3s_q2.json --workers 6 --device cpu --skip-existing

echo "===== FINAL: all-seeds CSV + summary CSV $(date) ====="
python experiments/collect_table.py --root $ROOT --out $ROOT/master_table_lr10_all_seeds.csv
python experiments/make_summary.py --root $ROOT --out $ROOT/master_table_lr10_summary.csv
echo "===== ALL DONE $(date) ====="
