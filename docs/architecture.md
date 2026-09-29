# ChangeForge 架构说明

作者：晨星

## 1. 分层与依赖方向

调用单向无环，**`core` 不反向依赖任何上层模块**：

```
        cli.py
          │
      pipeline (run / benchmark / tune)
          │
   ┌──────┼──────────┬───────────┐
   │      │          │           │
  data  detectors   eval        hpo ──(optuna)
   │      │          │           │
   └──────┴──────────┴───────────┘
                  │
                core   (types / errors / config / interfaces / validate)
```

- `core` 只依赖标准库 + numpy
- `hpo` 只依赖 `core` + optuna；目标函数由 `pipeline` 注入，避免反向依赖
- `pipeline` 对 `hpo` 为**延迟导入**（`tune()` 内部 import），避免启动即加载 optuna

## 2. 数据流

```
generate/load Dataset ──► validate(Dataset) ──► Detector.fit(signal).predict()
                                                        │
                                              detected change points
                                                        │
                                              evaluate(truth, pred, tolerance)
                                                        │
                                              EvalReport ──► BenchmarkRecord ──► benchmark.json
```

### 变化点索引语义（全局唯一约定）

变化点 `c` 表示**分段边界**：`signal[c]` 及其之后属于新的一段。

- 合法范围：`1 <= c <= n_samples - 1`，严格递增
- 例：n=10，变化点 `[3, 7]` → 段 `[0,3) [3,7) [7,10)`
- 该语义在 `core/types.py`、`data/synthetic.py`、`eval/metrics.py` 三处一致，任何新增模块必须沿用

## 3. 模块职责

| 模块 | 关键对象 | 说明 |
| --- | --- | --- |
| `core/types.py` | `Dataset` / `DetectionResult` / `EvalReport` / `ChangePoint` | 纯数据，字段均为可序列化标量（除 ndarray） |
| `core/errors.py` | `ChangeForgeError` + E1xx~E5xx | 统一 `code` + `context`，CLI 直接 `render()` |
| `core/config.py` | `Config` / `load_config` / `config_from_file` | 默认值 < `ENV_CHANGEFORGE_*` < 显式覆盖 |
| `core/interfaces.py` | `Detector` / `Evaluator` 等 Protocol | 鸭子类型契约，不强制继承 |
| `core/validate.py` | `validate_signal` / `validate_change_points` | 边界、NaN/Inf、段长校验 |
| `data/synthetic.py` | `generate_dataset` / `make_synthetic_suite` | 4 种变化类型 + 真值 |
| `data/loader.py` | `load_csv` / `load_npy` / `write_dataset_csv` | 无 pandas 依赖（标准库 csv） |
| `detectors/` | `register` / `build` / `list_detectors` | 注册表，**算法实现待接入** |
| `hpo/search.py` | `SearchSpace` / `run_hpo` / `HPOResult` | Optuna TPE，`n_jobs=1` |
| `eval/metrics.py` | `evaluate` / `match_change_points` / ARI | 纯 numpy，不引入 sklearn |
| `pipeline/pipeline.py` | `ChangeForgePipeline` | run / run_many / run_synthetic / tune |
| `pipeline/benchmark.py` | `benchmark` / `format_table` / `save_benchmark` | 固定列宽表格 + JSON 落盘 |

## 4. 错误码

| 段 | 含义 | 代表 |
| --- | --- | --- |
| E1xx | 配置与环境变量 | E101 取值非法 · E102 未知配置项 · E103 环境变量覆盖失败 |
| E2xx | 数据 | E201 信号非法 · E202 变化点非法 · E203 不存在 · E204 格式不支持 |
| E3xx | 检测器 | E301 未注册 · E302 未 fit · E303 超参非法 |
| E4xx | HPO | E401 study 失败 · E402 搜索空间非法 |
| E5xx | 流水线/评测/CLI | E501 评测失败 · E502 基准失败 · E503 CLI 参数错误 |

## 5. 扩展：新增一个检测器

1. 在 `changeforge/detectors/` 下新建模块，实现 `core.interfaces.Detector` 协议：
   `name` / `fit(signal)` / `predict() -> list[int]` / `get_params()` / `set_params()`
2. 用 `@register("my-detector", doc="一行说明")` 注册工厂函数
3. 在 `changeforge/detectors/__init__.py` 中 import 该模块（保证注册被执行）
4. 超参数校验失败时抛 `InvalidDetectorParamsError`（E303）
5. 不要把内部提示参数（verbose / 调试位）塞进 `**params` 传给估计器 —— 参数泄漏会导致 HPO 复现不一致

```python
from ..core.interfaces import Detector
from . import register


@register("my-detector", doc="示例检测器")
class MyDetector:
    name = "my-detector"

    def __init__(self, window: int = 20):
        self.window = window
        self._signal = None

    def fit(self, signal):
        self._signal = signal
        return self

    def predict(self) -> list[int]: ...
    def get_params(self):
        return {"window": self.window}

    def set_params(self, **params):
        unknown = set(params) - {"window"}
        if unknown:
            raise InvalidDetectorParamsError("未知参数", params=sorted(unknown))
        self.__dict__.update(params)
        return self
```

## 6. 配置

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `seed` | 42 | 合成数据与 HPO 采样器共用 |
| `n_jobs` | 1 | Windows 下强制钳制为 1（joblib 并发崩溃） |
| `tolerance` | 5 | 评测匹配容差（样本点） |
| `min_size` | 5 | 最小段长度 |
| `detector` | auto | 默认检测器名 |
| `hpo_n_trials` / `hpo_timeout_s` / `hpo_metric` | 20 / None / f1 | Optuna 参数 |
| `output_dir` | artifacts | 产物目录 |
| `strict` | True | 异常升级 vs 降级告警 |

## 7. 已知约束

- **numpy 2.x**：禁用 `np.trapz`（改用 `np.trapezoid`）、`np.float_`；数组判空一律 `is None`
- **Windows 并发**：joblib/sklearn 一律 `n_jobs=1`
- **sklearn 1.9**：多分类需显式 `solver="lbfgs"`，`multi_class` 参数已移除
- **无 pandas**：本环境未安装，CSV 走标准库
- **属性遮蔽**：report 类实例属性统一 `_report` 后缀，避免与方法同名
- **递归导入**：评测/指标模块内不要定义与 sklearn 导入别名同名的函数；如必须导入 sklearn，改名 `as _sk_xxx`
- **pytest 路径**：`pyproject.toml` 已设 `pythonpath=["."]`，干净环境可直接 `pytest`


## 8. 性能基线（quick 冒烟）

确定性可复现（`make_tier_series` 全走 `np.random.default_rng`，seed 固定）。
主指标 F1@tol_rel（口径 A：tol=0.01·n）；口径 C（tol=0.1·n/(K_true+1)，tol/段长恒 0.1）用于跨档可比。

| 档 | 检测器 | F1@A | F1@C | P | R | 时延ms |
|----|--------|------|------|----|----|----|
| D0 Smoke | numpy-pelt-l2 | 1.000 | 1.000 | 1.000 | 1.000 | 28.4 |
| D0 Smoke | numpy-binseg-l2 | 1.000 | 1.000 | 1.000 | 1.000 | 20.1 |
| D0 Smoke | numpy-window-l2 | 0.667 | 0.667 | 0.500 | 1.000 | 16.1 |
| D0 Smoke | numpy-mmd-rff | 1.000 | 1.000 | 1.000 | 1.000 | 475.4 |
| D0 Smoke | numpy-cusum | 0.417 | 0.417 | 0.267 | 1.000 | 0.7 |
| D0 Smoke | numpy-slidingt | 0.667 | 0.667 | 0.500 | 1.000 | 23.3 |
| D0 Smoke | ruptures-pelt-l2 | 1.000 | 1.000 | 1.000 | 1.000 | 3427.9 |
| D0 Smoke | ruptures-pelt-l1 | 1.000 | 1.000 | 1.000 | 1.000 | 4644.7 |
| D0 Smoke | ruptures-pelt-normal | 1.000 | 1.000 | 1.000 | 1.000 | 5433.8 |
| D0 Smoke | ruptures-pelt-rank | 1.000 | 1.000 | 1.000 | 1.000 | 5051.1 |
| D0 Smoke | ruptures-kernel-rbf | 1.000 | 1.000 | 1.000 | 1.000 | 21.6 |
| D0 Smoke | constab-cpd ⭐旗舰 | 1.000 | 1.000 | 1.000 | 1.000 | 93009.1 |
| D3 Medium | numpy-pelt-l2 | 0.333 | 0.500 | 0.500 | 0.250 | 21.7 |
| D3 Medium | numpy-binseg-l2 | 0.167 | 0.333 | 0.250 | 0.125 | 27.0 |
| D3 Medium | numpy-window-l2 | 0.000 | 0.000 | 0.000 | 0.000 | 13.8 |
| D3 Medium | numpy-mmd-rff | 0.000 | 0.200 | 0.000 | 0.000 | 1054.8 |
| D3 Medium | numpy-cusum | 0.444 | 0.556 | 0.400 | 0.500 | 0.7 |
| D3 Medium | numpy-slidingt | 0.000 | 0.000 | 0.000 | 0.000 | 19.4 |
| D3 Medium | ruptures-pelt-l2 | 0.333 | 0.500 | 0.500 | 0.250 | 3872.0 |
| D3 Medium | ruptures-pelt-l1 | 0.000 | 0.200 | 0.000 | 0.000 | 6547.9 |
| D3 Medium | ruptures-pelt-normal | 0.333 | 0.500 | 0.500 | 0.250 | 5801.3 |
| D3 Medium | ruptures-pelt-rank | 0.167 | 0.367 | 0.250 | 0.125 | 5807.2 |
| D3 Medium | ruptures-kernel-rbf | 0.000 | 0.200 | 0.000 | 0.000 | 17.5 |
| D3 Medium | constab-cpd ⭐旗舰 | 0.286 | 0.486 | 0.333 | 0.250 | 106634.4 |
| D4 Hard | numpy-pelt-l2 | 0.267 | 0.267 | 0.286 | 0.250 | 26.9 |
| D4 Hard | numpy-binseg-l2 | 0.167 | 0.167 | 0.250 | 0.125 | 44.6 |
| D4 Hard | numpy-window-l2 | 0.167 | 0.167 | 0.250 | 0.125 | 15.0 |
| D4 Hard | numpy-mmd-rff | 0.000 | 0.000 | 0.000 | 0.000 | 1140.0 |
| D4 Hard | numpy-cusum | 0.500 | 0.500 | 0.375 | 0.750 | 0.8 |
| D4 Hard | numpy-slidingt | 0.000 | 0.000 | 0.000 | 0.000 | 25.8 |
| D4 Hard | ruptures-pelt-l2 | 0.182 | 0.182 | 0.333 | 0.125 | 3318.4 |
| D4 Hard | ruptures-pelt-l1 | 0.222 | 0.222 | 1.000 | 0.125 | 5781.1 |
| D4 Hard | ruptures-pelt-normal | 0.400 | 0.400 | 1.000 | 0.250 | 6717.0 |
| D4 Hard | ruptures-pelt-rank | 0.222 | 0.222 | 1.000 | 0.125 | 6715.3 |
| D4 Hard | ruptures-kernel-rbf | 0.000 | 0.000 | 0.000 | 0.000 | 18.3 |
| D4 Hard | constab-cpd ⭐旗舰 | 0.000 | 0.000 | 0.000 | 0.000 | 107341.3 |
| D5 Expert | numpy-pelt-l2 | 0.526 | 0.526 | 0.556 | 0.500 | 65.7 |
| D5 Expert | numpy-binseg-l2 | 0.625 | 0.625 | 0.833 | 0.500 | 215.1 |
| D5 Expert | numpy-window-l2 | 0.686 | 0.629 | 0.800 | 0.600 | 28.7 |
| D5 Expert | numpy-mmd-rff | 0.182 | 0.182 | 1.000 | 0.100 | 3290.9 |
| D5 Expert | numpy-cusum | 0.407 | 0.407 | 0.324 | 0.550 | 1.4 |
| D5 Expert | numpy-slidingt | 0.389 | 0.389 | 0.438 | 0.350 | 42.1 |
| D5 Expert | ruptures-pelt-l2 | 0.606 | 0.606 | 0.769 | 0.500 | 7268.6 |
| D5 Expert | ruptures-pelt-l1 | 0.261 | 0.261 | 1.000 | 0.150 | 15674.1 |
| D5 Expert | ruptures-pelt-normal | 0.333 | 0.333 | 1.000 | 0.200 | 16628.0 |
| D5 Expert | ruptures-pelt-rank | 0.182 | 0.182 | 1.000 | 0.100 | 21941.0 |
| D5 Expert | ruptures-kernel-rbf | 0.095 | 0.095 | 1.000 | 0.050 | 72.1 |
| D5 Expert | constab-cpd ⭐旗舰 | 0.095 | 0.095 | 1.000 | 0.050 | 263153.8 |

零假设档 FPR（D1/D1b）控制在 α=0.05 附近；ConStab-CPD 的置换显著性闸门经 I12a 不变量校准
（conformal p = (1+#{null_max >= score})/(n_perm+1) <= α 才报警，不能用分位数阈值）。
完整结果与区分度自检见 `artifacts/benchmark.json` / `dgp_calibration.json`。
