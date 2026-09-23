# 自适应量化对照（2026-09-23）

新入口：`python -u -m experiments.adaptive_accuracy --seeds 0 1 2 3 4 --rounds 50`。
旧 matched_accuracy / lowbit 和量化函数保留，供历史复现；新实验默认并列运行
Original、SRK/Kashin × 1/2 bit × minmax-stochastic/optimized-uniform/lloyd-max。
Kashin 只用 balanced，D=65536，lambda=65536/50890。3/10 客户端、相同初始化、
客户端抽样和 mini-batch；不同模型训练后更新不同，跨轨迹 NMSE 不是同输入失真实验。

## 端点怎么算

每客户端每轮从自己的变换系数拟合，不用测试准确率、标签或跨轮调参。
optimized-uniform：在等距码本约束下最小化当前系数的经验 MSE。交替执行最近点分配，
以及最小二乘更新偏移和间距（两个自由参数）。从完整范围和五组分位区间启动，保留
实际误差最小的候选；允许端点内缩、尾部截断。多起点局部优化不保证全局最优。
lloyd-max：从优化均匀码本启动，用相邻中心的中点划分区域，再以区域均值更新中心，
空区域保留原中心。每轮最多30次迭代；保留不增加系数MSE的更新。

1 bit 两种方法都有两个自由重构值，因此使用相同优化结果，不把它们当独立方法优势。
2 bit 均匀码本四点等距，Lloyd-Max 四点可独立移动，因此后者自由度更大。
这里比较系数MSE；最终聚合NMSE和训练准确率未必同步改善，尤其Kashin是冗余框架。
两种新量化一般有偏。旧无偏min/max随机量化保留，差异同时涉及端点与舍入规则。
不能把新旧差异全部归因于端点，也不能把坐标平均误差当成统计无偏证明。
Kashin-balanced求解器仍优化系数范围；没有为新量化器重新优化表示，不宣称联合最优。

## 通信和更新

新方法显式生成索引和float32码本，解码仅使用索引和元数据，再逆变换、聚合并加到模型。
uniform: D*b+64 bit；Lloyd-Max: D*b+32*2**b bit。因此2bit Lloyd-Max多64bit/客户端/轮。
共享变换种子；不计下行、包头；是理论打包上行通信量，未实现真实网络bit打包。
Original: d*32 bit。每轮记录准确率、聚合NMSE、累计上行bit和压缩时间；diagnostics.json
记录逐客户端码本、系数MSE、客户端NMSE、变换与量化耗时。固定bit不是固定累计预算。

## 验证

`python -m unittest discover -s tests -p test_adaptive_quantization.py`
覆盖只凭索引和元数据重构、通信计费、常量/两值输入、目标不劣化及1bit一致性。
注入零量化输出后SRK/Kashin聚合更新必须为零，以检测绕过量化使用原始更新。
短轮实验仅用于通路验证，不作为50轮多种子收益结论。

理论参考：Lloyd, Least Squares Quantization in PCM, IEEE Transactions on Information
Theory, 1982. https://web.stanford.edu/class/ee398a/handouts/papers/Lloyd%20-%20Least%20Squares%20Q%20in%20PCM.pdf
