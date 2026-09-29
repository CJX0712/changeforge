"""ChangeForge 核心数据类型（dataclass）。

全局唯一约定 —— 变化点索引语义:

    变化点 ``c`` 表示分段边界：``signal[c]`` 及其之后属于新的一段。
    因此合法变化点满足 ``1 <= c <= n_samples - 1`` 且严格递增。

    样例：n=10，变化点 [3, 7] => 段为 [0,3) [3,7) [7,10)。

作者: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 信号数组：1D (n_samples,) 或 2D (n_samples, n_features)
Signal = np.ndarray


@dataclass(frozen=True, slots=True)
class ChangePoint:
    """单个变化点。

    Attributes:
        index: 变化点位置，满足 1 <= index <= n_samples - 1。
        score: 该变化点的显著度/代价下降，越大越显著；不可得时为 NaN。
        segment: 该点之后的段序号，从 0 开始；不可得时为 -1。
    """

    index: int
    score: float = float("nan")
    segment: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "score": self.score, "segment": self.segment}


@dataclass(slots=True, eq=False)
class Dataset:
    """一个待检测的信号及其（可选的）真值变化点。

    ``eq=False``：字段含 ndarray，默认生成的 ``__eq__`` 比较数组会返回
    数组而非布尔值（numpy 2.x 下会抛歧义错误），故关闭相等比较。
    """

    name: str
    signal: Signal
    change_points: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_features(self) -> int:
        return 1 if self.signal.ndim == 1 else int(self.signal.shape[1])

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.change_points)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_change_points": len(self.change_points),
            "has_ground_truth": self.has_ground_truth,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, eq=False)
class DetectionResult:
    """一次检测的输出。"""

    detector_name: str
    change_points: list[int]
    params: dict[str, Any] = field(default_factory=dict)
    scores: list[float] = field(default_factory=list)
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "change_points": list(self.change_points),
            "params": dict(self.params),
            "scores": list(self.scores),
            "elapsed_ms": self.elapsed_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """一次评测的输出（全部字段为纯标量，可直接序列化）。

    Attributes:
        n_true / n_pred: 真值/预测变化点个数。
        matched: 在容差内成功一对一匹配的预测点数。
        precision / recall / f1: 容差匹配下的分类指标。
        mean_abs_error: 已匹配点对的绝对索引误差均值；无匹配时为 NaN。
        count_error: n_pred - n_true（正数=过分割）。
        adjusted_rand_index: 分段划分的 ARI（越接近 1 越好）。
        tolerance: 本次评测使用的容差（样本点）。
    """

    n_true: int
    n_pred: int
    matched: int
    precision: float
    recall: float
    f1: float
    mean_abs_error: float
    count_error: int
    adjusted_rand_index: float
    tolerance: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_true": self.n_true,
            "n_pred": self.n_pred,
            "matched": self.matched,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_abs_error": self.mean_abs_error,
            "count_error": self.count_error,
            "adjusted_rand_index": self.adjusted_rand_index,
            "tolerance": self.tolerance,
        }
