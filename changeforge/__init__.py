"""ChangeForge —— 变化点检测（Changepoint Detection）工程框架。

作者: 晨星

分层约定（单向无环）::

    cli -> pipeline -> {data, hpo, detectors, eval} -> core

``core`` 不依赖上层任何模块，是所有模块的公共地基。
"""

from __future__ import annotations

from .core.config import Config, load_config
from .core.errors import ChangeForgeError
from .core.types import (
    ChangePoint,
    Dataset,
    DetectionResult,
    EvalReport,
)
from .pipeline.pipeline import ChangeForgePipeline, PipelineResult

# 算法实现导入即注册（numpy 兜底 + ruptures 后端 + ConStab 旗舰）。
# ruptures/skchange 的 import 全部惰性发生在工厂内部，离线环境不受影响。
from .detectors import numpy_impl as _numpy_impl  # noqa: F401
from .detectors import ruptures_impl as _ruptures_impl  # noqa: F401
from .ensemble import constab as _constab  # noqa: F401

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "ChangeForgeError",
    "ChangeForgePipeline",
    "ChangePoint",
    "Config",
    "Dataset",
    "DetectionResult",
    "EvalReport",
    "PipelineResult",
    "load_config",
    "__author__",
    "__version__",
]
