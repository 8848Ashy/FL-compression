# 本轮复现命令与实现说明

> 历史记录：下方命令针对重构前入口 `experiments.quantizer_study`。该脚本现已移至
> `archive/experiments/`，当前统一训练入口是 `python run_experiment.py`；下方旧命令不再是当前工作流。

在 D:\FL 中执行。解释器为 `D:\Miniconda\envs\fl-mnist\python.exe`，下方用 `$flPython`
简写。各输出目录必须不存在；脚本拒绝覆盖。完整训练只提供命令，本轮没有启动。

```powershell
$flPython = 'D:\Miniconda\envs\fl-mnist\python.exe'
& $flPython -m pytest tests -q

# 1. 生成独立校准轨迹：两seed、两轮、每轮三个完整更新。
& $flPython -u -m experiments.quantizer_study generate --seeds 100 101 --snapshots 1 2 --out results/quantizer_study_rerun/calibration_inputs

# 2. 校准并冻结两种表示各自的1/2bit码本。
& $flPython -u -m experiments.quantizer_study calibrate --inputs results/quantizer_study_rerun/calibration_inputs --sample-seed 2718 --samples-per-vector 2048 --out results/quantizer_study_rerun/calibration

# 3a. 优先复用经metadata SHA256验证的历史原始更新，记录原提交和客户端。
& $flPython -u -m experiments.quantizer_study reuse --source results/20260917_235716_matched_distortion --seeds 0 1 2 --snapshots 1 15 --out results/quantizer_study_rerun/evaluation_inputs

# 3b. 无历史文件时可改用可复现生成入口，不能同时写同一输出目录。
# & $flPython -u -m experiments.quantizer_study generate --seeds 0 1 2 --snapshots 1 2 --out results/quantizer_study_rerun/evaluation_inputs

# 4. 同输入失真/在线计时；随机基线每组8个draw，每次计时均预热后重复3次。
& $flPython -u -m experiments.quantizer_study evaluate --inputs results/quantizer_study_rerun/evaluation_inputs --codebooks results/quantizer_study_rerun/calibration/codebooks.json --draws 8 --timing-repeats 3 --out results/quantizer_study_rerun/evaluation

# 5. 只验证训练通路：一个seed、两轮；码本保持冻结。
& $flPython -u -m experiments.quantizer_study train --codebooks results/quantizer_study_rerun/calibration/codebooks.json --seeds 0 --rounds 2 --out results/quantizer_study_rerun/training

# 后续完整训练命令（本轮未运行；审核小实验报告后再执行）：
# & $flPython -u -m experiments.quantizer_study train --codebooks results/quantizer_study_rerun/calibration/codebooks.json --seeds 0 1 2 3 4 --rounds 50 --out results/quantizer_study_full/training
```

## 编码、解码和异常行为

新适配层 `compression/quantizer_payload.py` 的 `encode` 返回逻辑Payload，包含定长索引、
float32元数据。`decode`只接受Payload和可选共享码本。所有新研究分支都调用这个解码
结果做逆变换、聚合、服务器模型更新。Payload尚未进行物理bit打包，也没有网络传输。
表示方法、bits和方法标识由预先共享的实验配置确定，不作为逐向量额外字段发送。

支持float16/float32/float64输入，研究协议先转换到float32。无法用float32表示的非零值
或溢出值拒绝；空、非浮点、NaN/Inf输入拒绝。尺度用float64中间计算平方均值再取根，
保存并解码使用同一个float32尺度。非零RMS若不能表示为float32也拒绝。零向量尺度为0。
解码索引要求int64且在[0,2**bits)内；元数据要求float32、有限、一维，尺度非负，中心有序。

随机量化公共函数保留二元返回值和generator接口，只把原来的scale<1e-8透传改为真正
恒定时的常值解码；非零小范围正常随机舍入。double中间计算避免小范围下溢或大范围
溢出。合法数学概率没有被无提示裁剪替代。普通输入与旧数值可能有末位浮点差异。

逐向量C/D保留六起点、每起点最多30次和原停止条件。新增诊断可区分达到迭代上限、
中心不变、单一分配和Lloyd损失上升；不会把到达上限说成收敛。公共旧函数返回约定不变。
测试把现有C/D与旧提交351411e的实现逐张量比较，输出一致。

1bit C/D、E/F分别共用同一计算/训练路径。评估结果保留两种名称并标明alias，
训练结果用aliases字段表示，共21条独立模型路径（含Original），不是25项独立研究。

## 冻结码本与校准

每个完整更新独立做无中心化RMS归一化。每向量等量随机选2048个坐标，两种表示
使用同一坐标抽样，固定抽样seed2718；本轮12向量，每表示24576个拟合样本。
SRK与Kashin使用相同原始校准更新，分别拟合标准化码本。1bit共享两级结果；2bit
fixed-uniform用原有等距最小二乘拟合，fixed-lloyd-max从同一均匀初值继续原有Lloyd更新。
在线仅计算尺度、查共享码本和最近邻分配，没有再次拟合。预注册文件记录中心、
输入来源、配置、源码哈希、拟合诊断和SHA256；加载会核验文件内容，在线前后核验冻结。

保存文件包含20个float32中心，共640bit。实验采用预装假设；如单次分发整个中心库，
每个接收方额外至少640bit，10客户端共6400bit，另加配置、文件封装及传输成本。
JSON的实际字节数不等于中心净荷或每次理想上行通信量。

## 兼容与范围

保留旧matched_accuracy、matched_distortion、lowbit入口及历史结果。
adaptive_accuracy的压缩辅助函数接入新Payload，并保持旧位置参数和返回值；新增registry
可选参数。新的六量化器对照通过quantizer_study运行。没有自动改变旧入口默认方法。
Kashin-balanced仍最小化系数范围；本轮未更换求解器、lambda、模型、数据集，也未加入误差反馈。
本地工作区中的其他未提交文件保留；未commit、push、PR或远程实验。
