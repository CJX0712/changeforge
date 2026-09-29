"""数据载入：CSV / NPY / 内存数组 -> Dataset。

不依赖 pandas（本环境未安装），CSV 走标准库 ``csv``，保证最小依赖。

CSV 约定：
    首行为表头；除变化点标记列外的数值列都作为特征（多列即多变量信号）。
    变化点标记列（默认列名 ``is_change``，可用参数指定）中非 0 的行视为变化点。

作者: 晨星
"""

from __future__ import annotations

import csv
import os
from typing import Any, Sequence

import numpy as np

from ..core.errors import (
    DatasetNotFoundError,
    InvalidChangePointsError,
    InvalidSignalError,
    UnsupportedDataFormatError,
)
from ..core.types import Dataset, Signal

DEFAULT_CP_COLUMN = "is_change"
SUPPORTED_SUFFIXES = (".csv", ".npy")

__all__ = [
    "from_array",
    "load_csv",
    "load_dataset",
    "load_npy",
    "write_dataset_csv",
]


def from_array(
    values: Sequence[float] | Signal,
    change_points: Sequence[int] | None = None,
    name: str = "array",
    metadata: dict[str, Any] | None = None,
) -> Dataset:
    """把内存中的数组包装成 Dataset。

    Args:
        values: 1D 序列或 2D (n_samples, n_features) 结构。
        change_points: 真值变化点；None 表示无真值。
        name: 数据集名。
        metadata: 附加元信息。
    """
    if values is None:
        raise InvalidSignalError("values 不能为 None", name=name)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 2:
        raise InvalidSignalError("values 维度超过 2", name=name, ndim=array.ndim)
    points = list(change_points) if change_points is not None else []
    if points:
        n_samples = int(array.shape[0])
        bad = [p for p in points if not 1 <= int(p) <= n_samples - 1]
        if bad:
            raise InvalidChangePointsError(
                "真值变化点越界", name=name, invalid=bad, n_samples=n_samples
            )
    return Dataset(
        name=name,
        signal=array,
        change_points=[int(p) for p in points],
        metadata=dict(metadata or {}),
    )


def load_csv(
    path: str,
    value_column: str | Sequence[str] | None = None,
    change_point_column: str | None = DEFAULT_CP_COLUMN,
    name: str | None = None,
) -> Dataset:
    """从 CSV 载入信号。

    Args:
        path: CSV 文件路径。
        value_column: 作为特征的列名（单个或多个）；None 表示除标记列外的所有数值列。
        change_point_column: 变化点标记列名；None 表示没有真值列。
        name: 数据集名；None 时取文件名（不含后缀）。

    Raises:
        DatasetNotFoundError: 文件不存在。
        UnsupportedDataFormatError: 文件不是 CSV。
        InvalidSignalError: 无可用数值列或单元格无法解析。
    """
    if not os.path.isfile(path):
        raise DatasetNotFoundError("CSV 文件不存在", path=path)
    if not path.lower().endswith(".csv"):
        raise UnsupportedDataFormatError("不是 CSV 文件", path=path)

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise InvalidSignalError("CSV 为空（缺表头）", path=path) from exc
        # (原始行号, 行内容) —— 保留行号以便还原 0-based 样本索引
        rows = [
            (row_no, row)
            for row_no, row in enumerate(reader, start=1)
            if any(cell.strip() for cell in row)
        ]

    header = [column.strip() for column in header]
    if not rows:
        raise InvalidSignalError("CSV 无数据行", path=path)

    if value_column is None:
        feature_columns = [
            column
            for column in header
            if change_point_column is None or column != change_point_column
        ]
    elif isinstance(value_column, str):
        feature_columns = [value_column]
    else:
        feature_columns = list(value_column)

    missing = [column for column in feature_columns if column not in header]
    if missing:
        raise InvalidSignalError("CSV 缺少指定列", path=path, missing=missing)
    if change_point_column is not None and change_point_column not in header:
        change_point_column = None  # 标记列缺失 -> 视为无真值，不报错

    col_index = {column: idx for idx, column in enumerate(header)}
    values: list[list[float]] = []
    truth: list[int] = []
    for row_no, row in rows:
        try:
            values.append([float(row[col_index[column]]) for column in feature_columns])
        except (ValueError, IndexError) as exc:
            raise InvalidSignalError(
                "CSV 单元格无法解析为浮点数", path=path, row=row_no
            ) from exc
        if change_point_column is not None:
            cell = row[col_index[change_point_column]].strip()
            if cell not in {"", "0", "0.0", "false", "False"}:
                # 数据行 row_no 从 1 开始（表头已弹出），样本索引为 0-based
                truth.append(row_no - 1)

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.reshape(-1)
    dataset_name = name or os.path.splitext(os.path.basename(path))[0]
    return Dataset(name=dataset_name, signal=signal, change_points=truth)


def load_npy(
    path: str,
    change_points: Sequence[int] | None = None,
    name: str | None = None,
) -> Dataset:
    """从 .npy 载入数组；真值变化点由参数给出。"""
    if not os.path.isfile(path):
        raise DatasetNotFoundError("NPY 文件不存在", path=path)
    if not path.lower().endswith(".npy"):
        raise UnsupportedDataFormatError("不是 NPY 文件", path=path)
    array = np.load(path)
    return from_array(
        array,
        change_points=change_points,
        name=name or os.path.splitext(os.path.basename(path))[0],
    )


def load_dataset(path: str, **kwargs: Any) -> Dataset:
    """按后缀分派载入；不支持的后缀抛 UnsupportedDataFormatError。"""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return load_csv(path, **kwargs)
    if suffix == ".npy":
        return load_npy(path, **kwargs)
    raise UnsupportedDataFormatError(
        "不支持的数据格式", path=path, supported=list(SUPPORTED_SUFFIXES)
    )


def write_dataset_csv(dataset: Dataset, path: str) -> str:
    """把 Dataset 写出为 CSV（含 ``is_change`` 标记列），便于外部复现。"""
    signal = np.asarray(dataset.signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    truth = set(int(c) for c in dataset.change_points)
    n_features = signal.shape[1]
    header = [f"x{idx}" for idx in range(n_features)] + [DEFAULT_CP_COLUMN]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_idx in range(signal.shape[0]):
            row = [f"{value:.10g}" for value in signal[row_idx]]
            row.append("1" if row_idx in truth else "0")
            writer.writerow(row)
    return path
