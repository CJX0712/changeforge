"""公共校验工具（core 层，所有上层模块复用）。

注意 numpy 2.x 语义：判断"数组是否为空/是否给定"一律用 ``is None``，
不要写 ``if arr else ...``（数组真值判断会抛 ValueError）。

作者: 晨星
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import InvalidChangePointsError, InvalidSignalError
from .types import Dataset

__all__ = ["validate_change_points", "validate_dataset", "validate_signal"]


def validate_signal(signal: Any, min_samples: int = 2, name: str = "signal") -> np.ndarray:
    """校验并归一化信号数组。

    Returns:
        转成 float64 的数组（1D 或 2D）。

    Raises:
        InvalidSignalError: 非数组 / 空 / 维度不支持 / 含 NaN 或 Inf / 样本数不足。
    """
    if signal is None:
        raise InvalidSignalError("信号不能为 None", name=name)
    if not isinstance(signal, np.ndarray):
        try:
            array = np.asarray(signal, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidSignalError("信号无法转换为数值数组", name=name) from exc
    else:
        array = signal
    if array.size == 0:
        raise InvalidSignalError("信号为空", name=name)
    if array.ndim not in (1, 2):
        raise InvalidSignalError(
            "信号维度只支持 1D 或 2D (n_samples, n_features)", name=name, ndim=array.ndim
        )
    if array.shape[0] < min_samples:
        raise InvalidSignalError(
            "样本数不足", name=name, n_samples=array.shape[0], min_samples=min_samples
        )
    if not np.issubdtype(array.dtype, np.number):
        array = array.astype(np.float64, copy=False)
    else:
        array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise InvalidSignalError("信号含 NaN 或 Inf", name=name)
    return array


def validate_change_points(
    change_points: Any,
    n_samples: int,
    min_size: int = 1,
    name: str = "change_points",
) -> list[int]:
    """校验变化点索引并升序去重返回。

    合法条件：整数、``1 <= c <= n_samples - 1``、严格递增、相邻间距（含两端）
    不小于 ``min_size``。

    Raises:
        InvalidChangePointsError: 任一条件不满足。
    """
    if change_points is None:
        return []
    if isinstance(change_points, np.ndarray):
        raw = change_points.tolist()
    elif isinstance(change_points, (list, tuple)):
        raw = list(change_points)
    else:
        raise InvalidChangePointsError(
            "变化点必须是 list/tuple/ndarray", name=name, type=type(change_points).__name__
        )
    if n_samples < 2:
        raise InvalidChangePointsError("样本数不足以容纳变化点", name=name, n_samples=n_samples)
    points: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            if isinstance(item, float) and float(item).is_integer():
                item = int(item)
            else:
                raise InvalidChangePointsError("变化点必须是整数", name=name, value=item)
        value = int(item)
        if not 1 <= value <= n_samples - 1:
            raise InvalidChangePointsError(
                "变化点越界（应在 1 ~ n_samples-1）",
                name=name,
                value=value,
                n_samples=n_samples,
            )
        points.append(value)
    if len(set(points)) != len(points):
        raise InvalidChangePointsError("变化点存在重复", name=name)
    points.sort()
    bounds = [0, *points, n_samples]
    for left, right in zip(bounds[:-1], bounds[1:]):
        if right - left < min_size:
            raise InvalidChangePointsError(
                "段长度小于 min_size", name=name, min_size=min_size, length=right - left
            )
    return points


def validate_dataset(
    dataset: Dataset, min_size: int = 1, require_truth: bool = False
) -> Dataset:
    """就地校验 Dataset：信号合法 +（可选）真值变化点合法。返回同一对象。"""
    if not isinstance(dataset, Dataset):
        raise InvalidSignalError("期望 Dataset 实例", type=type(dataset).__name__)
    dataset.signal = validate_signal(dataset.signal, name=dataset.name)
    dataset.change_points = validate_change_points(
        dataset.change_points, dataset.n_samples, min_size=min_size, name=f"{dataset.name}.cp"
    )
    if require_truth and not dataset.change_points:
        raise InvalidChangePointsError("该数据集缺少真值变化点", name=dataset.name)
    return dataset


def is_finite_float(value: Any) -> bool:
    """工具：判断标量是否为有限浮点数（NaN/Inf 视为 False）。"""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
