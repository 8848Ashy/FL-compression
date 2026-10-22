# Fourier Kashin Frame Ablation

当前独立消融项目只保留 Fourier Kashin frame，用于研究真实联邦学习更新的系数峰值和量化误差。

Fourier frame 使用 FFT 和随机相位，不构造稠密矩阵，不依赖 Hadamard/FWHT，也不要求维度或冗余维度是 2 的幂，因此可以使用任意 `D`，适合大规模 FL 模型维度。

运行单一真实更新实验：

```text
python kashin_ablation/frame_benchmark.py
```

输入为 `results/sample_delta.pt`。当前配置为 `d=50890`、`D=101780`、2 bit、10 次 Kashin 迭代。结果写入 `results/frame_comparison.csv` 和 `results/frame_comparison.md`。
