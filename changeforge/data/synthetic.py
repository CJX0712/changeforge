"""合成数据生成（含真值变化点）。

支持四种变化类型：

    constant  分段常数 + 高斯噪声（均值漂移检测的标准基线）
    variance  均值恒定、分段方差切换（方差变化检测）
    trend     分段线性趋势，斜率切换且段间连续
    mixed     均值 + 方差同时切换

所有生成器都返回 :class:`~changeforge.core.types.Dataset`，其中
``change_points`` 为**真值**变化点，可直接用于评测。

变化点索引语义见 ``core/types.py``：``signal[c]`` 及其之后属于新的一段。

作者: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import InvalidChangePointsError
from ..core.types import Dataset, Signal

SYNTHETIC_KINDS: tuple[str, ...] = ("constant", "variance", "trend", "mixed")

__all__ = [
    "SYNTHETIC_KINDS",
    "generate_dataset",
    "make_synthetic_suite",
]


def _segment_bounds(
    rng: np.random.Generator, n_samples: int, n_change_points: int, min_size: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """随机切分出 n_change_points 个变化点，保证每段长度 >= min_size。

    Returns:
        (change_points, segments)，segments 为左闭右开区间列表。
    """
    n_segments = n_change_points + 1
    slack = n_samples - n_segments * min_size
    if slack < 0:
        raise InvalidChangePointsError(
            "样本数不足以切分出该数量的变化点",
            n_samples=n_samples,
            n_change_points=n_change_points,
            min_size=min_size,
        )
    # 把剩余长度随机分配给各段（多项式分布保证总和恰为 slack）
    probs = np.full(n_segments, 1.0 / n_segments)
    extra = rng.multinomial(slack, probs)
    lengths = np.full(n_segments, min_size, dtype=np.int64) + extra.astype(np.int64)
    change_points: list[int] = []
    segments: list[tuple[int, int]] = []
    cursor = 0
    for idx, length in enumerate(lengths):
        start, end = cursor, cursor + int(length)
        segments.append((start, end))
        cursor = end
        if idx < n_change_points:
            change_points.append(int(end))
    return change_points, segments


def _draw_means(
    rng: np.random.Generator, n_segments: int, mean_range: tuple[float, float], min_shift: float
) -> np.ndarray:
    """为每段抽样均值，保证相邻段均值差 >= min_shift（最多重试 20 次）。"""
    low, high = mean_range
    means = np.zeros(n_segments, dtype=np.float64)
    means[0] = rng.uniform(low, high)
    for idx in range(1, n_segments):
        for _ in range(20):
            candidate = rng.uniform(low, high)
            if abs(candidate - means[idx - 1]) >= min_shift:
                means[idx] = candidate
                break
        else:  # 重试耗尽：直接取反向偏移，保证可分辨
            direction = -1.0 if means[idx - 1] > (low + high) / 2 else 1.0
            means[idx] = float(np.clip(means[idx - 1] + direction * min_shift * 2.0, low, high))
    return means


def _draw_stds(
    rng: np.random.Generator, n_segments: int, std_range: tuple[float, float]
) -> np.ndarray:
    low, high = std_range
    return rng.uniform(low, high, size=n_segments).astype(np.float64)


def _draw_slopes(
    rng: np.random.Generator, n_segments: int, slope_range: tuple[float, float]
) -> np.ndarray:
    low, high = slope_range
    slopes = rng.uniform(low, high, size=n_segments).astype(np.float64)
    # 避免相邻段斜率过于接近，交替符号增强可检测性
    for idx in range(1, n_segments):
        if abs(slopes[idx] - slopes[idx - 1]) < (high - low) * 0.1:
            slopes[idx] = -slopes[idx]
    return slopes


def generate_dataset(
    kind: str = "constant",
    n_samples: int = 1000,
    n_change_points: int = 3,
    noise_std: float = 0.3,
    seed: int | None = None,
    min_size: int = 30,
    mean_range: tuple[float, float] = (-3.0, 3.0),
    std_range: tuple[float, float] = (0.2, 1.5),
    slope_range: tuple[float, float] = (-0.05, 0.05),
    name: str | None = None,
) -> Dataset:
    """生成一条带真值变化点的合成信号。

    Args:
        kind: constant / variance / trend / mixed，见模块 docstring。
        n_samples: 信号长度。
        n_change_points: 真值变化点个数。
        noise_std: 观测噪声标准差（叠加在段信号之上）。
        seed: 随机种子；None 表示不可复现。
        min_size: 每段最小长度。
        mean_range / std_range / slope_range: 各类型的段参数抽样范围。
        name: 数据集名；None 时自动生成。

    Returns:
        Dataset，signal 为 float64 的 1D 数组，change_points 为真值。

    Raises:
        InvalidChangePointsError: 参数不足以切分出要求的段数。
    """
    if kind not in SYNTHETIC_KINDS:
        raise InvalidChangePointsError(
            "不支持的合成类型", kind=kind, supported=list(SYNTHETIC_KINDS)
        )
    if n_samples < 2 or n_change_points < 0:
        raise InvalidChangePointsError(
            "n_samples 必须 >= 2 且 n_change_points >= 0",
            n_samples=n_samples,
            n_change_points=n_change_points,
        )
    rng = np.random.default_rng(seed)
    change_points, segments = _segment_bounds(rng, n_samples, n_change_points, min_size)
    n_segments = len(segments)

    signal: Signal = np.zeros(n_samples, dtype=np.float64)
    means = _draw_means(rng, n_segments, mean_range, min_shift=1.0)
    stds = _draw_stds(rng, n_segments, std_range)
    slopes = _draw_slopes(rng, n_segments, slope_range)

    baseline = float(means[0]) if kind in {"constant", "variance", "mixed"} else 0.0
    for seg_idx, (start, end) in enumerate(segments):
        length = end - start
        if kind in {"constant", "variance", "mixed"}:
            level = float(means[seg_idx])
        else:
            level = baseline
        if kind == "variance":
            base = np.zeros(length, dtype=np.float64)
            scale = float(stds[seg_idx])
        elif kind == "trend":
            step = np.arange(length, dtype=np.float64)
            base = level + slopes[seg_idx] * step
            baseline = float(base[-1]) + float(slopes[seg_idx])
            scale = 1.0
        elif kind == "mixed":
            base = np.full(length, level, dtype=np.float64)
            scale = float(stds[seg_idx])
        else:  # constant
            base = np.full(length, level, dtype=np.float64)
            scale = 1.0
        noise = rng.normal(0.0, noise_std, size=length) if noise_std > 0 else np.zeros(length)
        signal[start:end] = base + scale * noise

    metadata: dict[str, Any] = {
        "kind": kind,
        "seed": seed,
        "noise_std": noise_std,
        "min_size": min_size,
        "n_segments": n_segments,
        "segment_lengths": [end - start for start, end in segments],
        "means": means.tolist(),
        "stds": stds.tolist() if kind in {"variance", "mixed"} else [],
        "slopes": slopes.tolist() if kind == "trend" else [],
        "synthetic": True,
    }
    dataset_name = name or f"syn-{kind}-{n_samples}-{n_change_points}cp-seed{seed}"
    return Dataset(
        name=dataset_name,
        signal=signal,
        change_points=change_points,
        metadata=metadata,
    )


def make_synthetic_suite(
    seed: int = 42,
    n_samples: int = 1000,
    kinds: tuple[str, ...] | None = None,
    n_change_points: int = 3,
) -> list[Dataset]:
    """生成一组标准合成数据集，用于基准测试与演示。

    每个 kind 使用 ``seed + kind_index`` 作为种子，保证可复现且彼此不同。
    """
    kinds = kinds or SYNTHETIC_KINDS
    datasets: list[Dataset] = []
    for offset, kind in enumerate(kinds):
        datasets.append(
            generate_dataset(
                kind=kind,
                n_samples=n_samples,
                n_change_points=n_change_points,
                noise_std=0.3,
                seed=seed + offset,
                min_size=max(10, n_samples // (n_change_points + 1) // 3),
            )
        )
    return datasets
