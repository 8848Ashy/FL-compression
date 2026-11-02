# 服务器部署指南（enine @ 172.31.100.235）

从零到跑通批量实验，约 15 分钟（依赖网速）。

## 1. 获取代码

```bash
git clone git@github.com:8848Ashy/FL-compression.git ~/FL-compression
cd ~/FL-compression
git checkout refactor-drive-style
```

（若服务器没有 GitHub 私钥权限：在本地 `git bundle create fl.bundle --all`，scp 上去后
`git clone fl.bundle ~/FL-compression`。）

## 2. 环境（Python 3.10，venv 隔离）

```bash
python3 -m venv ~/venvs/fl
source ~/venvs/fl/bin/activate
pip install -r requirements.txt        # torch 2.14.1 CPU 版约 190MB
```

## 3. 数据（2.3GB，校园网 scp 一分钟级）

本地（Windows PowerShell）：
```powershell
scp -r D:\FL\data hczhang@172.31.100.235:~/FL-compression/
```

> torchvision 的 EMNIST 源（biometrics.nist.gov）直连经常失败，不建议靠 download=True。

## 4. 冒烟验证（2 轮，约 1 分钟）

```bash
cd ~/FL-compression
source ~/venvs/fl/bin/activate
python run_experiment.py --task emnist_cnn --partition dirichlet --dirichlet_alpha 0.2 \
  --num_clients 100 --clients_per_round 5 --total_rounds 2 --seed 42 \
  --transform fourier_kashin_balanced --bits 1 --lam 1.3 --stochastic 1 \
  --kashin_iters 15 --local_epochs 3 --server_lr 1.25 \
  --output_dir results/_smoke --exp_name smoke --device cpu
```

## 5. 批量跑（40 核并行，16 workers）

```bash
nohup python experiments/run_parallel.py \
  --jobs experiments/jobs_stage4_mid_3seeds.json \
  --workers 16 --device cpu > results_stage4_mid_3seeds.out 2>&1 &
```

- 每个实验子进程限 2 线程（OMP_NUM_THREADS=2），16 并发 ≈ 32 核，不干扰其他用户；
- 进度：`tail -f results_stage4_mid_3seeds.out`，单组日志在 `results/stage4_mid_3seeds/logs/`；
- 预计 33 组（10 配置 × 3 seeds + hadamard×3）约 30~50 分钟（本机串行需 8 小时）；
- **必须 `--device cpu`**：否则 auto 会抢 2 张 RTX 6000D，且 GPU 结果与本地 seed-42 数据不可比。

## 6. 结果取回

```powershell
scp -r hczhang@172.31.100.235:~/FL-compression/results/stage4_mid_3seeds D:\FL\results\
```
