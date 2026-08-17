# 联邦学习 Kashin 压缩实验 - 项目交接说明

更新时间：2026-08-14

## 1. 项目目标

在 MNIST 联邦学习中研究通信压缩。当前重点是比较：

- Original：无压缩 FedAvg；
- SRK：Hadamard 随机旋转 + 随机 k 级量化；
- Kashin：随机冗余紧框架 + Kashin 系数求解 + 随机 k 级量化。

研究问题是：**在相同上行通信预算下，Kashin 是否比 SRK 有更小的性能损失？**

不要预设 Kashin 一定更好；当前结果并未稳定支持该结论。

## 2. 当前主代码与环境

- 主代码：`D:\FL\2017.py`
- Conda 环境：`fl_env`
- 数据集：MNIST
- 模型：MLP，`784 -> 64 -> 10`
- 客户端数：10
- 每个客户端训练样本数：600
- 测试集：MNIST 测试集前 1000 张
- 本地训练默认：2 epoch、SGD、学习率 0.05

## 3. 重要理论背景（给 AI 的约束）

### 3.1 为什么上传更新量 Delta-w

当前新管线压缩客户端更新量：

`delta_w_i = w_local_i - w_global`

服务器更新：

`w_global_next = w_global + average_i(Q(delta_w_i))`

在无压缩、客户端等权重时，这与直接平均完整本地模型参数完全等价；但压缩时上传更新量更合理，因为它只传递本轮新增信息。

### 3.2 SRK 与 Kashin 的差异

- SRK：将长度 `d` 的更新量补零到 `d_pow2` 后，进行随机符号 Hadamard 旋转，再量化；Hadamard 是等维可逆旋转。
- Kashin：使用冗余紧框架 `U`，求长度 `D>d` 的系数 `a`，使 `delta_w ≈ U a`；`kashin_solve` 通过迭代截断寻找峰值较小、更均匀的系数，再量化 `a`。
- Kashin 不是简单“换一个随机矩阵”；关键是“冗余框架 + 迭代系数求解”。

当前 Kashin frame 是随机部分 Hadamard + 随机列符号：

`U = sqrt(D/d) * R * H_D * S`

其中 `R` 为随机选行，`H_D` 为归一化 Hadamard，`S` 为随机正负号对角矩阵。

## 4. 已完成的代码工作

### 4.1 更新量工具与验证

已经新增：

- `state_dict_subtract(local_state, global_state)`
- `state_dict_add(global_state, delta_state)`
- `local_train_delta(global_model, dataloader, epochs=2, lr=0.05)`
- `federated_round_original_update(global_model, client_deltas)`

已验证：

- 完整模型平均 vs 平均 Delta-w 后加回全局模型：最大差异约 `7.45e-09`；
- 贴近主循环的另一验证：最大差异约 `1.86e-09`；
- 两者均通过阈值 `1e-6`。

### 4.2 SRK 更新量版本

已新增：

`federated_round_srk_update(global_model, client_deltas, k_levels, rotation_seed=None)`

特点：

- 压缩 Delta-w，不再压缩完整模型；
- 同一轮所有客户端与服务器共享同一个随机符号向量；
- 使用向量化 `_fwht_fast`，数学上已验证与旧 `fast_walsh_hadamard_transform` 等价；
- SRK 管线结构验证已经通过，无 NaN/Inf、维度正确。

### 4.3 Kashin 更新量版本

已新增：

`federated_round_kashin_update(global_model, client_deltas, k_levels, frame, iterations=10)`

特点：

- 压缩 Delta-w；
- 使用共享 `KashinFrame`，不按客户端或轮次重新创建；
- 量化前调用 `kashin_solve`；
- Kashin 管线结构验证已经通过，无 NaN/Inf、维度正确。

### 4.4 Kashin 单元测试

已有 `KashinFrame`、`kashin_solve`、`kashin_unit_test`。

已测：

- 原始参数维度：`d = 50890`
- 框架系数维度：`D = 65536`
- 冗余度：`D/d ≈ 1.2878`
- canonical 重构误差约 `1.46e-07`
- Kashin 重构误差约 `1.46e-07`
- 20 个随机向量中，Kashin/naive 系数峰值比值中位数约 `0.90`。

注意：这只证明当前测试中峰值有约 10% 下降和重构正确；不证明 Kashin 在联邦准确率上必然优于 SRK。

## 5. 当前主实验开关

`2017.py` 顶部当前配置应为：

```python
RUN_FULL_EXPERIMENT = True
FOCUS_ON_KASHIN = True
FIXED_BITS = 2
FIXED_K_LEVELS = 2 ** FIXED_BITS
NUM_ROUNDS_FOCUS = 8
TRADEOFF_BITS = [1, 2, 3]
TRADEOFF_ROUNDS = 8
```

运行主程序后：

1. 先执行四个轻量验证；
2. 验证都通过后，运行 `run_focus_kashin_experiment(timestamp)`；
3. 默认只跑 Original、SRK、Kashin 三种 Delta-w 方法；
4. 不跑旧 SK/SVK/五算法实验。

## 6. 当前通信量口径（极重要）

统一口径为：**每客户端、每原始参数维度的上行 bit 数**。

- Original：`32.0`
- SRK：

`(d_pow2 * ceil(log2(k)) + 64) / d`

- Kashin：

`(D * ceil(log2(k)) + 64) / d`

当前 `d=50890`，`d_pow2=D=65536`，所以 SRK 与 Kashin 同一 `k` 下通信量严格相同。

例如 `b=2`，即 `k=2**b=4`：

- 单轮通信量：约 `2.576852 bit/dim/client`
- 若运行 8 轮：累计约 `20.6148 bit/dim/client`

因此 SRK 与 Kashin 是公平对比对象。Original 的 32 bit/dim 仅作为无压缩性能参考，不是同通信量基线。

## 7. 当前图与正确解读

### 图 1：固定 bit 的收敛曲线

文件格式：

`kashin_vs_srk_accuracy_<timestamp>.png`

设置：固定 `b=2`（即 `k=4`），8 轮。

包含 Original、SRK、Kashin 的准确率随联邦轮次变化。

正确表述：

> 单次运行中，SRK、Kashin 可以接近无压缩 Original。若某一轮压缩方法略高于 Original，通常只能视为随机量化/训练噪声造成的波动，不能宣称压缩优于无压缩。

### 图 2：通信-错误率 trade-off 图

最新示例文件：

`D:\FL\kashin_vs_srk_tradeoff_20260814_000902.png`

设置：`b=1,2,3`，每个点训练 **8 轮**。

- 横轴：8 轮累计通信量；越左越省通信；
- 纵轴：最终分类错误率；越低越好；
- 同一个 `b` 下 SRK 与 Kashin 横坐标相同，谁的点更低谁更好；
- 棕色虚线：Original 的无压缩错误率参考。

最新单次结果大致为：

| b | 累计 bit/dim/client | SRK error | Kashin error | 单次较好者 |
|---|---:|---:|---:|---|
| 1 | 10.31 | 9.8% | 10.3% | SRK |
| 2 | 20.61 | 10.7% | 11.0% | SRK |
| 3 | 30.92 | 9.7% | 10.0% | SRK |

正确结论：

> 在当前低冗余度 `lambda≈1.29`、单随机种子、8轮的设置下，SRK 略优于 Kashin；差距约 0.3-0.5 个百分点。不能宣称 Kashin 已取得优势。

重要问题：最新 trade-off 图的左上角文字仍误写成 `4 federated rounds per point`，但横轴与实际配置已是 8 轮。若修改图，必须把这行改成 `8 federated rounds per point`。标题也更准确应为 `Trade-off: Communication vs Classification Error`，因为纵轴是 error 不是 accuracy。

### 为什么 bit 越多不一定单调更好

每个 b 都是独立短训练；模型初始化、DataLoader 打乱、随机量化和随机旋转都会造成波动。因此不应依据当前单种子、三个点的曲线声称“bit 越多误差一定越小”。

## 8. 当前汇报可用表述

> 我将通信对象从完整模型参数改为客户端本地更新量 Delta-w，并验证了无压缩时它与原 FedAvg 数值等价。随后实现了 SRK 与 Kashin 的更新量压缩。为公平比较，二者均使用 65536 个量化系数，因此同一 bit 下通信量严格相同。当前在 MNIST/MLP 上进行了单次 8 轮实验。结果显示二者均接近无压缩基线；在目前低冗余度设置下，SRK 略优于 Kashin。下一步可研究 Kashin 冗余度、截断迭代次数和相同总通信预算下的配置，但当前不应过度宣称 Kashin 优势。

## 9. 后续建议（不要未经用户同意直接大改）

优先级从高到低：

1. 修正 trade-off 图文字：`4 federated rounds per point` 改为 `8 federated rounds per point`，标题改为 Classification Error；
2. 若老师要求进一步探索：在固定低 bit 下，分别测试 Kashin `iterations=5,10,20`；
3. 再测试更高冗余度，例如 `D=131072`，对应 `lambda≈2.58`；
4. 高冗余度会提高通信量，必须与 SRK 比较相同总 bit 预算，不能仍使用相同 b；
5. 若需要强结论，再做多个随机种子并报告均值/标准差；当前用户暂不需要该工作。

## 10. 行为约束

- 用户希望分步骤下命令给 VS Code ClaudeCode/DeepSeek 执行；不要一次给出大规模重构任务。
- 每次先完成、运行轻量验证，再进入下一步。
- 用户当前目标是能理解并向老师汇报，不是在写正式论文；解释应使用中文、简洁、避免过度理论化。
- 不要为了得到“Kashin 更好”的结果而选择性调参或夸大单次实验。
