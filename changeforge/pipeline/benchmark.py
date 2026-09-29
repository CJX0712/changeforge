"""基准测试：多数据集 x 多检测器 -> 指标表 + JSON 落盘。

作者: 晨星
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..core.config import Config, load_config
from ..core.errors import BenchmarkError
from ..core.types import Dataset, EvalReport
from ..core.validate import validate_dataset
from .pipeline import ChangeForgePipeline

# 表格列定义：(表头, 宽度, 取值属性)。dataset 列放宽以免名字被截断错位。
COLUMNS: tuple[tuple[str, int, str], ...] = (
    ("dataset", 26, "dataset"),
    ("detector", 18, "detector"),
    ("n", 10, "n_samples"),
    ("n_true", 10, "n_true"),
    ("n_pred", 10, "n_pred"),
    ("prec", 10, "precision"),
    ("recall", 10, "recall"),
    ("f1", 10, "f1"),
    ("mae", 10, "mean_abs_error"),
    ("ari", 10, "adjusted_rand_index"),
    ("ms", 10, "elapsed_ms"),
    ("status", 10, "status"),
)

__all__ = ["BenchmarkRecord", "benchmark", "format_table", "save_benchmark"]


def _fmt_number(value: float) -> str:
    if value is None:
        return "-"
    number = float(value)
    if number != number:  # NaN
        return "-"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    return f"{number:.3f}"


@dataclass(slots=True, eq=False)
class BenchmarkRecord:
    """一条基准结果（一行）。"""

    dataset: str
    detector: str
    n_samples: int = 0
    n_true: int = 0
    n_pred: int = 0
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    mean_abs_error: float = float("nan")
    adjusted_rand_index: float = float("nan")
    tolerance: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        dataset: Dataset,
        detector: str,
        change_points: Sequence[int],
        report: EvalReport | None,
        elapsed_ms: float,
        tolerance: int,
        params: Mapping[str, Any] | None = None,
    ) -> BenchmarkRecord:
        return cls(
            dataset=dataset.name,
            detector=detector,
            n_samples=dataset.n_samples,
            n_true=len(dataset.change_points),
            n_pred=len(change_points),
            precision=report.precision if report else float("nan"),
            recall=report.recall if report else float("nan"),
            f1=report.f1 if report else float("nan"),
            mean_abs_error=report.mean_abs_error if report else float("nan"),
            adjusted_rand_index=report.adjusted_rand_index if report else float("nan"),
            tolerance=tolerance,
            elapsed_ms=elapsed_ms,
            params=dict(params or {}),
        )

    @classmethod
    def failure(cls, dataset: Dataset, detector: str, error: BaseException) -> BenchmarkRecord:
        return cls(
            dataset=dataset.name,
            detector=detector,
            n_samples=dataset.n_samples,
            n_true=len(dataset.change_points),
            status="error",
            error=f"{type(error).__name__}: {error}",
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "detector": self.detector,
            "n_samples": self.n_samples,
            "n_true": self.n_true,
            "n_pred": self.n_pred,
            "tolerance": self.tolerance,
            "status": self.status,
            "params": dict(self.params),
        }
        for key in (
            "precision",
            "recall",
            "f1",
            "mean_abs_error",
            "adjusted_rand_index",
            "elapsed_ms",
        ):
            value = getattr(self, key)
            payload[key] = None if value != value else float(value)  # NaN -> None
        if self.error:
            payload["error"] = self.error
        return payload


def benchmark(
    datasets: Sequence[Dataset],
    detectors: Sequence[str],
    config: Config | None = None,
    params_by_detector: Mapping[str, Mapping[str, Any]] | None = None,
    tolerance: int | None = None,
    verbose: bool = True,
) -> list[BenchmarkRecord]:
    """在 datasets x detectors 的笛卡尔积上跑基准。

    Args:
        datasets: 数据集列表（须含真值，否则该条记为无评测）。
        detectors: 检测器名列表。
        config: 配置；None 时自动加载。
        params_by_detector: ``{检测器名: {参数...}}``。
        tolerance: 覆盖 config.tolerance。
        verbose: 逐条打印进度。

    Returns:
        BenchmarkRecord 列表（失败条目 status="error"，其余 "ok"）。
    """
    config = config or load_config()
    if not datasets:
        raise BenchmarkError("datasets 为空")
    if not detectors:
        raise BenchmarkError("detectors 为空")
    tol = int(tolerance if tolerance is not None else config.tolerance)
    pipeline = ChangeForgePipeline(config=config)
    params_map: Mapping[str, Mapping[str, Any]] = params_by_detector or {}
    records: list[BenchmarkRecord] = []

    for data in datasets:
        dataset = validate_dataset(data, min_size=config.min_size)
        for detector_name in detectors:
            params = dict(params_map.get(detector_name, {}))
            start = time.perf_counter()
            try:
                result = pipeline.run(
                    dataset=dataset,
                    detector=detector_name,
                    params=params,
                    tolerance=tol,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                record = BenchmarkRecord.from_result(
                    dataset=dataset,
                    detector=detector_name,
                    change_points=result.change_points,
                    report=result.evaluation_report,
                    elapsed_ms=elapsed_ms,
                    tolerance=tol,
                    params=result.params,
                )
            except Exception as exc:  # noqa: BLE001 - 基准需要收集失败而非中断
                if config.strict and isinstance(exc, KeyboardInterrupt):
                    raise
                warnings.warn(
                    f"{dataset.name} / {detector_name} 失败: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                record = BenchmarkRecord.failure(dataset, detector_name, exc)
            records.append(record)
            if verbose:
                state = "OK " if record.status == "ok" else "ERR"
                score = _fmt_number(record.f1)
                print(f"[{state}] {record.dataset} / {record.detector} f1={score}")
    return records


def format_table(records: Sequence[BenchmarkRecord]) -> str:
    """把基准结果渲染成固定列宽的文本表（对齐用 ``f"{value:<width}"``）。"""
    if not records:
        return "(无基准记录)"
    header = "".join(f"{name:<{width}}" for name, width, _ in COLUMNS)
    lines = [header, "-" * len(header)]
    for record in records:
        cells: list[str] = []
        for _, width, attr in COLUMNS:
            value = getattr(record, attr, "")
            if isinstance(value, float):
                text = _fmt_number(value)
            elif isinstance(value, int):
                text = str(value)
            else:
                text = str(value)
            if len(text) > width - 1:
                text = text[: width - 1]
            cells.append(f"{text:<{width}}")
        lines.append("".join(cells))
    return "\n".join(lines)


def save_benchmark(
    records: Sequence[BenchmarkRecord],
    path: str | None = None,
    config: Config | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """把基准结果写成 JSON（含生成时间与配置快照）。返回实际路径。"""
    config = config or load_config()
    target = path or os.path.join(config.output_dir, "benchmark.json")
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "0.1.0",
        "author": "晨星",
        "config": config.to_dict(),
        "n_records": len(records),
        "records": [record.to_dict() for record in records],
    }
    if extra:
        payload.update(dict(extra))
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target
