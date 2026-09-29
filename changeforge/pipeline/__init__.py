"""流水线层：端到端执行 + 基准测试。

依赖方向：``pipeline -> {data, hpo, detectors, eval} -> core``。
``hpo`` 为延迟导入（optuna 较重），只有真正调参时才加载。

作者: 晨星
"""

from __future__ import annotations

from .benchmark import (
    BenchmarkRecord,
    benchmark,
    format_table,
    save_benchmark,
)
from .pipeline import ChangeForgePipeline, PipelineResult

__all__ = [
    "BenchmarkRecord",
    "ChangeForgePipeline",
    "PipelineResult",
    "benchmark",
    "format_table",
    "save_benchmark",
]
