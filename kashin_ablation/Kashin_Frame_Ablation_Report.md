# Kashin Frame Ablation Report

## Objective

本项目用于验证 Fourier Kashin frame 在真实联邦学习更新上的系数幅度和量化误差表现。它是独立诊断实验，不修改主联邦学习程序。

## Method

Fourier frame 通过随机相位、FFT 和抽取/嵌入操作实现紧框架分析与合成。实现不保存 `d × D` 稠密矩阵，不使用 Hadamard/FWHT，因此 `D` 可以是任意整数。Kashin solver 对分析系数反复截断、合成并更新残差。

当前正式实验设置：真实 FL update，`d=50890`，`D=101780`，2 bit，10 次迭代，随机种子 2026。

## Results

运行 `python kashin_ablation/frame_benchmark.py` 后，结果保存在：

- `results/frame_comparison.csv`
- `results/frame_comparison.md`

正式结果只报告 Fourier frame，不把未完成的其他候选方法作为质量对比结论。

## Recommendation

当前项目保留 Fourier Kashin frame 作为后续主 FL 实验的候选实现。它的优势是任意 `D`、不需要稠密矩阵、并且适合大模型维度；这不等同于已经证明它在所有指标上优于其他未完成候选方法。

## Discarded Experiment

Pure random frame was explored as a candidate, but the full `d=50890` benchmark was not completed because its matrix-vector operations were too computationally expensive. It is not included in the formal quality comparison.
