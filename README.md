# ChangeForge

**变化点检测（Changepoint Detection）工程框架** —— 从合成数据生成到检测、评测、超参搜索、基准落盘的一体化流水线。

- 作者：晨星
- 域：时间序列 / 信号的变化点检测（均值漂移、方差切换、趋势切换）
- 技术栈：Python 3.13 · numpy · scipy · scikit-learn · ruptures · Optuna · ruff · pytest
- 状态：**算法与基准已产出**（12 个检测器 + ConStab-CPD 旗舰，纯 numpy 离线兜底 + ruptures 后端）


---

## 快速开始

```bash
# 1) 创建虚拟环境并安装（Windows 用 .venv\Scripts\python.exe）
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .

# 2) 冒烟：查看环境与配置
python -m changeforge.cli info

# 3) 端到端演示（落盘 artifacts/benchmark.json）
python examples/run_demo.py

# 4) 跑测试与静态检查
python -m pytest tests -q
python -m ruff check .
```

Docker：

```bash
docker build -t changeforge:0.1.0 .
docker run --rm changeforge:0.1.0 info
```

## 命令行

| 子命令 | 作用 |
| --- | --- |
| `info` | 打印版本、配置、环境、已注册检测器 |
| `list-detectors` | 列出已注册检测器 |
| `list-datasets` | 列出合成数据类型 |
| `gen` | 生成合成数据并导出 CSV（含 `is_change` 真值列） |
| `run` | 单数据集端到端：检测 + 评测（`--json` 输出结构化结果） |
| `benchmark` | 多数据集 × 多检测器基准，落盘 `benchmark.json` |
| `tune` | Optuna 调参（`--space` 指定搜索空间 JSON） |

配置可通过环境变量覆盖，前缀 `ENV_CHANGEFORGE_`，例如：

```bash
ENV_CHANGEFORGE_SEED=7 ENV_CHANGEFORGE_TOLERANCE=12 python -m changeforge.cli info
```

优先级：默认值 < 环境变量 < 显式参数/`--config` 配置文件。

## 目录结构

```
changeforge/                 # 仓库根
├── changeforge/             # 包
│   ├── core/                # 类型 · 错误码(E100~E500) · 配置 · 协议 · 校验
│   ├── data/                # 合成数据生成（含真值）+ CSV/NPY 载入
│   ├── detectors/           # 变化点检测器注册表（算法模块待接入）
│   ├── hpo/                 # Optuna 调参
│   ├── eval/                # 评测指标（容差匹配 P/R/F1 · MAE · ARI）
│   ├── pipeline/            # ChangeForgePipeline.run() + benchmark()
│   └── cli.py               # argparse 入口
├── examples/run_demo.py     # 端到端演示
├── tests/                   # pytest
├── docs/architecture.md     # 架构说明
└── Dockerfile / Makefile / pyproject.toml / requirements*.txt
```

架构与扩展方式见 [docs/architecture.md](docs/architecture.md)。

## 性能基准（quick 冒烟，确定性可复现）

> 口径：主指标 **F1@tol_rel**（tol_rel = 0.01·n）；β = lam·d·σ̂²·log n，σ̂ = MAD·1.4826，lam=3（BIC 类默认）。
> 旗舰 **ConStab-CPD** 用 n_perm=19 循环块置换显著性闸门（conformal p≤α 才报警）；
> quick 档为降速版（减少每档序列数），完整数字见 `artifacts/benchmark.json`，区分度自检见 `artifacts/dgp_calibration.json`。

### 检测能力（非空档 · F1 / P / R / 单次时延 ms）

| 档 | 检测器 | F1 | P | R | 时延ms |
|----|--------|----|----|----|----|
| D0 Smoke | numpy-pelt-l2 | 1.000 | 1.000 | 1.000 | 24.4 |
| D0 Smoke | numpy-binseg-l2 | 1.000 | 1.000 | 1.000 | 18.2 |
| D0 Smoke | numpy-window-l2 | 0.667 | 0.500 | 1.000 | 15.0 |
| D0 Smoke | numpy-mmd-rff | 1.000 | 1.000 | 1.000 | 372.8 |
| D0 Smoke | numpy-cusum | 0.417 | 0.267 | 1.000 | 0.9 |
| D0 Smoke | numpy-slidingt | 0.667 | 0.500 | 1.000 | 24.9 |
| D0 Smoke | ruptures-pelt-l2 | 1.000 | 1.000 | 1.000 | 3193.0 |
| D0 Smoke | ruptures-pelt-l1 | 1.000 | 1.000 | 1.000 | 4238.2 |
| D0 Smoke | ruptures-pelt-normal | 1.000 | 1.000 | 1.000 | 4829.5 |
| D0 Smoke | ruptures-pelt-rank | 1.000 | 1.000 | 1.000 | 4672.6 |
| D0 Smoke | ruptures-kernel-rbf | 1.000 | 1.000 | 1.000 | 19.0 |
| D0 Smoke | constab-cpd ⭐旗舰 | 1.000 | 1.000 | 1.000 | 85351.0 |
| D3 Medium | numpy-pelt-l2 | 0.333 | 0.500 | 0.250 | 24.8 |
| D3 Medium | numpy-binseg-l2 | 0.167 | 0.250 | 0.125 | 26.8 |
| D3 Medium | numpy-window-l2 | 0.000 | 0.000 | 0.000 | 14.2 |
| D3 Medium | numpy-mmd-rff | 0.000 | 0.000 | 0.000 | 1080.5 |
| D3 Medium | numpy-cusum | 0.444 | 0.400 | 0.500 | 0.6 |
| D3 Medium | numpy-slidingt | 0.000 | 0.000 | 0.000 | 19.3 |
| D3 Medium | ruptures-pelt-l2 | 0.333 | 0.500 | 0.250 | 4012.3 |
| D3 Medium | ruptures-pelt-l1 | 0.000 | 0.000 | 0.000 | 6688.2 |
| D3 Medium | ruptures-pelt-normal | 0.333 | 0.500 | 0.250 | 6002.6 |
| D3 Medium | ruptures-pelt-rank | 0.167 | 0.250 | 0.125 | 5712.8 |
| D3 Medium | ruptures-kernel-rbf | 0.000 | 0.000 | 0.000 | 26.8 |
| D3 Medium | constab-cpd ⭐旗舰 | 0.286 | 0.333 | 0.250 | 135521.8 |
| D4 Hard | numpy-pelt-l2 | 0.261 | 0.200 | 0.375 | 25.2 |
| D4 Hard | numpy-binseg-l2 | 0.111 | 0.100 | 0.125 | 91.2 |
| D4 Hard | numpy-window-l2 | 0.353 | 0.333 | 0.375 | 14.7 |
| D4 Hard | numpy-mmd-rff | 0.000 | 0.000 | 0.000 | 1105.8 |
| D4 Hard | numpy-cusum | 0.500 | 0.350 | 0.875 | 1.2 |
| D4 Hard | numpy-slidingt | 0.000 | 0.000 | 0.000 | 22.3 |
| D4 Hard | ruptures-pelt-l2 | 0.222 | 0.200 | 0.250 | 3236.7 |
| D4 Hard | ruptures-pelt-l1 | 0.000 | 0.000 | 0.000 | 7305.6 |
| D4 Hard | ruptures-pelt-normal | 0.222 | 1.000 | 0.125 | 6544.3 |
| D4 Hard | ruptures-pelt-rank | 0.000 | 0.000 | 0.000 | 8199.1 |
| D4 Hard | ruptures-kernel-rbf | 0.000 | 0.000 | 0.000 | 19.8 |
| D4 Hard | constab-cpd ⭐旗舰 | 0.000 | 0.000 | 0.000 | 139270.0 |
| D5 Expert | numpy-pelt-l2 | 0.762 | 0.727 | 0.800 | 97.2 |
| D5 Expert | numpy-binseg-l2 | 0.778 | 0.875 | 0.700 | 481.9 |
| D5 Expert | numpy-window-l2 | 0.718 | 0.737 | 0.700 | 58.4 |
| D5 Expert | numpy-mmd-rff | 0.095 | 1.000 | 0.050 | 6980.1 |
| D5 Expert | numpy-cusum | 0.340 | 0.296 | 0.400 | 1.7 |
| D5 Expert | numpy-slidingt | 0.537 | 0.524 | 0.550 | 59.5 |
| D5 Expert | ruptures-pelt-l2 | 0.821 | 0.842 | 0.800 | 11450.7 |
| D5 Expert | ruptures-pelt-l1 | 0.647 | 0.786 | 0.550 | 30502.0 |
| D5 Expert | ruptures-pelt-normal | 0.429 | 0.750 | 0.300 | 29905.9 |
| D5 Expert | ruptures-pelt-rank | 0.333 | 1.000 | 0.200 | 29151.3 |
| D5 Expert | ruptures-kernel-rbf | 0.095 | 1.000 | 0.050 | 73.3 |
| D5 Expert | constab-cpd ⭐旗舰 | 0.095 | 1.000 | 0.050 | 734778.4 |

### 零假设档（D1 / D1b · 每序列假阳率 FPR）

| 档 | 检测器 | FPR |
|----|--------|-----|
| D1 Null-iid | numpy-pelt-l2 | 0.00 |
| D1 Null-iid | numpy-binseg-l2 | 0.00 |
| D1 Null-iid | numpy-window-l2 | 0.00 |
| D1 Null-iid | numpy-mmd-rff | 0.00 |
| D1 Null-iid | numpy-cusum | 0.80 |
| D1 Null-iid | numpy-slidingt | 0.00 |
| D1 Null-iid | ruptures-pelt-l2 | 0.00 |
| D1 Null-iid | ruptures-pelt-l1 | 0.00 |
| D1 Null-iid | ruptures-pelt-normal | 0.00 |
| D1 Null-iid | ruptures-pelt-rank | 0.00 |
| D1 Null-iid | ruptures-kernel-rbf | 0.00 |
| D1 Null-iid | constab-cpd | 0.00 |
| D1b Null-AR | numpy-pelt-l2 | 1.00 |
| D1b Null-AR | numpy-binseg-l2 | 1.00 |
| D1b Null-AR | numpy-window-l2 | 1.00 |
| D1b Null-AR | numpy-mmd-rff | 0.00 |
| D1b Null-AR | numpy-cusum | 1.00 |
| D1b Null-AR | numpy-slidingt | 1.00 |
| D1b Null-AR | ruptures-pelt-l2 | 1.00 |
| D1b Null-AR | ruptures-pelt-l1 | 0.00 |
| D1b Null-AR | ruptures-pelt-normal | 0.20 |
| D1b Null-AR | ruptures-pelt-rank | 0.00 |
| D1b Null-AR | ruptures-kernel-rbf | 0.00 |
| D1b Null-AR | constab-cpd | 0.00 |

### 区分度自检（dgp_calibration）

- D0 Smoke: · best_single=1.000 · spread=0.583 · smoke_min_f1=0.417 · 融合增益=+0.000 · GateB回退=否
- D1 Null-iid: · 融合增益=+0.000 · GateB回退=否 · FPR=[numpy-pelt-l2=0.00, numpy-binseg-l2=0.00, numpy-window-l2=0.00, numpy-mmd-rff=0.00, numpy-cusum=0.80, numpy-slidingt=0.00, ruptures-pelt-l2=0.00, ruptures-pelt-l1=0.00, ruptures-pelt-normal=0.00, ruptures-pelt-rank=0.00, ruptures-kernel-rbf=0.00, constab-cpd=0.00]
- D1b Null-AR: · 融合增益=+0.000 · GateB回退=否 · FPR=[numpy-pelt-l2=1.00, numpy-binseg-l2=1.00, numpy-window-l2=1.00, numpy-mmd-rff=0.00, numpy-cusum=1.00, numpy-slidingt=1.00, ruptures-pelt-l2=1.00, ruptures-pelt-l1=0.00, ruptures-pelt-normal=0.20, ruptures-pelt-rank=0.00, ruptures-kernel-rbf=0.00, constab-cpd=0.00]
- D3 Medium: · best_single=0.444 · spread=0.444 · 融合增益=-0.159 · GateB回退=是
- D4 Hard: · best_single=0.500 · spread=0.500 · 融合增益=-0.500 · GateB回退=是
- D5 Expert: · best_single=0.821 · spread=0.725 · 融合增益=-0.725 · GateB回退=是

### oracle-K 上界参考（仅供对照，禁止进主表）

- D0 Smoke: ruptures-dynp-l2 (K=1) F1=1.000 —— oracle-K 上界，非公平对比
- D3 Medium: ruptures-dynp-l2 (K=4) F1=0.375 —— oracle-K 上界，非公平对比
- D4 Hard: ruptures-dynp-l2 (K=8) F1=0.125 —— oracle-K 上界，非公平对比
- D5 Expert: ruptures-dynp-l2 (K=20) F1=0.800 —— oracle-K 上界，非公平对比

## 状态与 Roadmap

- [x] 工程骨架：core / data / hpo / eval / pipeline / cli
- [x] 合成数据（constant / variance / trend / mixed）+ 真值变化点
- [x] 评测指标与基准落盘（JSON）
- [x] Optuna 调参基建与 CLI
- [x] 算法模块接入（12 个检测器 + ConStab-CPD 旗舰，纯 numpy 离线兜底 + ruptures 后端）
- [x] 基准数据集与真实性能数字（quick 冒烟，见上）
- [ ] 结果可视化

## 许可证

Apache-2.0 · 作者 晨星
