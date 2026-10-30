#!/usr/bin/env bash
# ==============================================================================
# 联邦学习压缩实验总控脚本 (experiments.sh)
# 作用：统一配置公共参数，并按阶段批量调用 run_experiment.py 运行对比实验。
#
# 常用运行方式（在终端执行）：
#   1. 默认参数运行：
#      bash experiments.sh stage2_bpd_law_doscale
#   2. 临时覆盖轮数/客户端数/输出目录（适合快速测试）：
#      TOTAL_ROUNDS=20 CLIENTS=50 PER_ROUND=10 OUT=./results/test_run bash experiments.sh stage2_bpd_law_doscale
#   3. 跑完后画图（指向同一个 OUT 目录）：
#      python plot_results.py --results_dir ./results/drive_style --output_dir ./results/drive_style
# ==============================================================================

# 严格模式：
# -e: 任何一条命令报错立即停止脚本
# -u: 使用未定义的变量时报错
# -o pipefail: 管道命令中有任何一环失败则整体视为失败
set -euo pipefail

# ==============================================================================
# 一、 全局超参数配置（可直接修改冒号后面的默认数字，也可在命令行临时覆盖）
# ==============================================================================

# 联邦通信总轮数（默认 100 轮；平时调试可改为 20，正式跑可改为 100~200）
TOTAL_ROUNDS=${TOTAL_ROUNDS:-50}

# 参与联邦学习的客户端总数 N（默认把数据集切分给 100 个客户端）
CLIENTS=${CLIENTS:-100}

# 每一轮随机抽取参与训练并上传更新的客户端数量 M（默认每轮抽 10 个客户端求平均）
PER_ROUND=${PER_ROUND:-10}

# 实验结果输出总目录（生成的每个实验子文件夹、hparams.json、metrics.csv 都会存放在这里）
# 【提示】如果你想每次运行自动加上“日期+时间”防止覆盖旧结果，可以把下面这行改成：
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT=${OUT:-./results/${TIMESTAMP}_${1:-drive_style}}


# 公共命令行参数数组（会传递给下面所有的 python run_experiment.py 调用）
# 【手动修改任务与数据分布指南】：
# 1. --task 可选值：
#      mnist_mlp  : 10分类 MNIST + 全连接网络（当前默认，速度最快但任务较简单）
#      mnist_cnn  : 10分类 MNIST + 2层卷积网络 CNN
#      emnist_cnn : 62分类 EMNIST + 2层卷积网络 CNN（DRIVE论文同款，最能拉开不同算法准确率差距）
# 2. --partition 可选值：
#      iid        : 独立同分布（每个客户端数据分布均匀）
#      dirichlet  : Non-IID 异构分布（需在后面加上 --dirichlet_alpha 0.3，数值越小异构越严重）
#    例如改成 EMNIST + Non-IID 时，将第一行改为：
#    COMMON=(--task emnist_cnn --partition dirichlet --dirichlet_alpha 0.3 --num_clients "$CLIENTS"
COMMON=(--task emnist_cnn --partition dirichlet --dirichlet_alpha 0.3 --num_clients "$CLIENTS"
        --clients_per_round "$PER_ROUND" --total_rounds "$TOTAL_ROUNDS"
        --output_dir "$OUT")

# ==============================================================================
# 二、 第一阶段实验（Stage 1）：固定 1-bit 下准确率随通信轮数的收敛对比 (对应 Exp0)
# ==============================================================================

# 1. 有缩放因子 (doscale) 的 1-bit 对比实验
# 每个客户端上传 1-bit 符号的同时，附带上传 1 个浮点数缩放因子 scale（如 L1均值 E[|x|]）
stage1_doscale() {
  # 依次对比 4 种变换方法：
  # - uncompressed   : 32-bit 不压缩基准线 (FedAvg)
  # - raw            : 不做旋转变换，直接对原始梯度做 1-bit 量化 (即 SignSGD + scale)
  # - hadamard       : 随机符号 + 快速 Walsh-Hadamard 旋转后再 1-bit 量化 (等价于 MMSE 版 DRIVE)
  # - fourier_kashin : 你的 Fourier Kashin 紧框架分解后再 1-bit 量化
  for transform in uncompressed raw hadamard fourier_kashin; do
    python run_experiment.py "${COMMON[@]}" --transform "$transform" \
      --scale_mode doscale \
      --bits 1 \
      --lam 1.29 \
      --exp_name "stage1_doscale_${transform}"
  done
  # 注：原版 DRIVE 在数学上等价于上面的 hadamard + doscale + bits=1，因此无需重复跑。
}

# 2. 无缩放因子 (noscale) 的 1-bit 对比实验
# 客户端不上传任何浮点数 scale，仅依靠预设的静态截断边界 clip_val 进行 1-bit 量化
stage1_noscale() {
  for transform in raw hadamard fourier_kashin; do
    python run_experiment.py "${COMMON[@]}" --transform "$transform" \
      --scale_mode noscale \
      --noscale_bound_mode dimension_scaled \
      --clip_val 0.25 \
      --bits 1 \
      --lam 1.29 \
      --exp_name "stage1_noscale_${transform}"
      # 参数说明：
      # --noscale_bound_mode dimension_scaled : 根据展开维度 D 自动按 1/sqrt(D) 缩放各变换的理论截断界
      # --clip_val 0.25                       : 基准 L2 范数尺度参数
  done
}

# ==============================================================================
# 三、 第二阶段实验（Stage 2）：不同每维度比特数 (BPD = bits * lam) 下的准确率定律 (对应 Exp1)
# ==============================================================================

# 1. 无缩放因子 (noscale) 模式下的 BPD 网格扫描实验
# 扫描比特数 bits in {1, 2, 3} 与 Kashin 冗余度 lam in {1.15, 1.3, 1.6, 2.0}
stage2_bpd_law() {
  # 外层循环：遍历每个系数的量化比特数 b = 1, 2, 3
  for bits in 1 2 3; do

    # (a) 跑 Raw (实际 BPD = bits * 1.0) 和 Hadamard (因补零到 2^k，实际 BPD ≈ bits * 1.29)
    for transform in raw hadamard; do
      python run_experiment.py "${COMMON[@]}" --transform "$transform" \
        --scale_mode noscale --noscale_bound_mode dimension_scaled \
        --clip_val 0.25 --bits "$bits" --lam 1.3 \
        --exp_name "stage2_noscale_${transform}_b${bits}"
    done

    # (b) 跑你的 Fourier Kashin：遍历不同冗余度 lam = D/d (实际 BPD = bits * lam)
    for lam in 1 1.15 1.3 1.5 1.8 2.0; do
      python run_experiment.py "${COMMON[@]}" --transform fourier_kashin \
        --scale_mode noscale --noscale_bound_mode dimension_scaled \
        --clip_val 0.25 --bits "$bits" --lam "$lam" \
        --exp_name "stage2_noscale_fourier_kashin_b${bits}_lam${lam}"
    done
  done
}

# 2. 有缩放因子 (doscale) 模式下的 BPD 网格扫描实验（★最推荐用来验证 Kashin 压低无穷范数的优势）
# 在此模式下：bits=1 使用 L1 均值缩放；bits>1 使用真实无穷范数 ||a||_inf 作为均匀量化区间边界
stage2_bpd_law_doscale() {
  # 外层循环：遍历每个系数的量化比特数 b = 1, 2, 3
  for bits in 1 2 3; do

    # (a) 跑 Raw 和 Hadamard 在 b = 1, 2, 3 下的表现
    for transform in raw hadamard; do
      python run_experiment.py "${COMMON[@]}" --transform "$transform" \
        --scale_mode doscale --bits "$bits" --lam 1.3 \
        --exp_name "stage2_doscale_${transform}_b${bits}"
    done

    # (b) 跑 Fourier Kashin 在不同 (bits, lam) 组合下的表现
    for lam in 1 1.15 1.3 1.5 1.8 2.0; do
      python run_experiment.py "${COMMON[@]}" --transform fourier_kashin \
        --scale_mode doscale --bits "$bits" --lam "$lam" \
        --exp_name "stage2_doscale_fourier_kashin_b${bits}_lam${lam}"
    done
  done
}

# ==============================================================================
# 四、 命令行入口路由：根据你在 `bash experiments.sh <阶段名>` 传入的第 1 个参数执行对应函数
# ==============================================================================
case "${1:-}" in
  stage1_doscale)         stage1_doscale ;;
  stage1_noscale)         stage1_noscale ;;
  stage2_bpd_law)         stage2_bpd_law ;;
  stage2_bpd_law_doscale) stage2_bpd_law_doscale ;;
  *)
    echo "Usage: $0 {stage1_doscale|stage1_noscale|stage2_bpd_law|stage2_bpd_law_doscale}" >&2
    exit 2
    ;;
esac