"""ChangeForge 的 Protocol 接口契约。

用 ``typing.Protocol`` 描述上层模块必须实现的形状，避免为抽象基类引入
不必要的继承耦合（检测器实现方只要"鸭子类型"满足即可）。

作者: 晨星
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import Dataset, DetectionResult, EvalReport


@runtime_checkable
class SupportsFitPredict(Protocol):
    """最小 fit/predict 契约。"""

    def fit(self, signal: np.ndarray) -> SupportsFitPredict: ...
    def predict(self) -> list[int]: ...


@runtime_checkable
class Detector(Protocol):
    """变化点检测器契约（detectors/ 下所有实现必须满足）。"""

    name: str

    def fit(self, signal: np.ndarray) -> Detector:
        """在信号上拟合；返回 self 以支持链式调用。"""
        ...

    def predict(self) -> list[int]:
        """返回变化点索引列表，严格递增且满足 1 <= c <= n_samples - 1。"""
        ...

    def get_params(self) -> dict[str, Any]:
        """返回当前超参数字典（用于基准记录与复现）。"""
        ...

    def set_params(self, **params: Any) -> Detector:
        """就地设置超参数；未知参数必须抛 InvalidDetectorParamsError。"""
        ...


@runtime_checkable
class DataGenerator(Protocol):
    """合成数据生成器契约（data/synthetic.py 实现）。"""

    def generate(self) -> Dataset: ...


@runtime_checkable
class Evaluator(Protocol):
    """评测器契约（eval/metrics.py 实现）。"""

    def evaluate(self, dataset: Dataset, predicted: list[int]) -> EvalReport: ...


@runtime_checkable
class PipelineLike(Protocol):
    """流水线契约（pipeline/pipeline.py 实现）。"""

    def run(self, dataset: Dataset, detector: str | None = None) -> Any: ...


__all__ = [
    "DataGenerator",
    "Detector",
    "Dataset",
    "DetectionResult",
    "EvalReport",
    "Evaluator",
    "PipelineLike",
    "SupportsFitPredict",
]
