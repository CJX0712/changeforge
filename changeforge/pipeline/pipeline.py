"""ChangeForgePipeline：数据 -> 检测 -> 评测 的端到端执行。

作者: 晨星
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.config import Config, load_config
from ..core.errors import DetectorNotFoundError, PipelineError
from ..core.types import Dataset, DetectionResult, EvalReport
from ..core.validate import validate_dataset
from ..data import from_array, generate_dataset
from ..detectors import build, list_detectors
from ..eval import evaluate

__all__ = ["ChangeForgePipeline", "PipelineResult"]


@dataclass(slots=True, eq=False)
class PipelineResult:
    """一次流水线执行的完整产物（可直接 JSON 序列化）。"""

    dataset_name: str
    detector_name: str
    change_points: list[int]
    params: dict[str, Any] = field(default_factory=dict)
    scores: list[float] = field(default_factory=list)
    elapsed_ms: float = 0.0
    evaluation_report: EvalReport | None = None
    dataset_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def f1(self) -> float:
        return float("nan") if self.evaluation_report is None else self.evaluation_report.f1

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "detector_name": self.detector_name,
            "change_points": list(self.change_points),
            "params": dict(self.params),
            "scores": list(self.scores),
            "elapsed_ms": self.elapsed_ms,
            "evaluation_report": (
                None if self.evaluation_report is None else self.evaluation_report.to_dict()
            ),
            "dataset_summary": dict(self.dataset_summary),
        }


class ChangeForgePipeline:
    """端到端流水线。

    Example:
        >>> pipeline = ChangeForgePipeline()
        >>> result = pipeline.run(signal=my_signal, detector="my-detector")
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config: Config = config or load_config()
        self._history: list[PipelineResult] = []

    @property
    def config(self) -> Config:
        return self._config

    @property
    def history(self) -> list[PipelineResult]:
        return list(self._history)

    def available_detectors(self) -> list[str]:
        return list_detectors()

    def _resolve_dataset(
        self,
        dataset: Dataset | None,
        signal: Sequence[float] | np.ndarray | None,
        true_change_points: Sequence[int] | None,
        name: str,
    ) -> Dataset:
        if dataset is not None and signal is not None:
            raise PipelineError("dataset 与 signal 只能二选一")
        if dataset is None:
            if signal is None:
                raise PipelineError("必须提供 dataset 或 signal")
            dataset = from_array(signal, change_points=true_change_points, name=name)
        return validate_dataset(dataset, min_size=self._config.min_size)

    def run(
        self,
        dataset: Dataset | None = None,
        signal: Sequence[float] | np.ndarray | None = None,
        true_change_points: Sequence[int] | None = None,
        detector: str | None = None,
        params: dict[str, Any] | None = None,
        tolerance: int | None = None,
        with_evaluation: bool = True,
        dataset_name: str = "inline",
    ) -> PipelineResult:
        """执行 检测（+ 评测）。

        Args:
            dataset: 已构造的数据集；与 signal 二选一。
            signal: 裸信号数组；与 dataset 二选一。
            true_change_points: signal 模式下的真值变化点。
            detector: 检测器名；None 取 ``config.detector``。
            params: 检测器超参数。
            tolerance: 评测容差；None 取 ``config.tolerance``。
            with_evaluation: 是否有真值可评测（无真值时自动跳过）。

        Raises:
            DetectorNotFoundError: 检测器未注册（算法模块尚未接入）。
        """
        data = self._resolve_dataset(dataset, signal, true_change_points, dataset_name)
        detector_name = str(detector or self._config.detector).strip()
        if not list_detectors():
            raise DetectorNotFoundError(
                "检测器注册表为空：算法模块尚未接入（Phase 2b）",
                detector=detector_name,
            )
        model = build(detector_name, **dict(params or {}))

        start = time.perf_counter()
        model.fit(data.signal)
        detected = self._coerce_points(getattr(model, "predict")(), data.n_samples)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detection = DetectionResult(
            detector_name=detector_name,
            change_points=detected,
            params=dict(model.get_params()) if hasattr(model, "get_params") else {},
            scores=list(getattr(model, "scores_", []) or []),
            elapsed_ms=elapsed_ms,
        )

        report: EvalReport | None = None
        if with_evaluation and data.change_points:
            report = evaluate(
                data.change_points,
                detection.change_points,
                n_samples=data.n_samples,
                tolerance=int(tolerance if tolerance is not None else self._config.tolerance),
            )
        elif with_evaluation:
            warnings.warn(
                f"数据集 {data.name} 无真值变化点，跳过评测",
                RuntimeWarning,
                stacklevel=2,
            )

        result = PipelineResult(
            dataset_name=data.name,
            detector_name=detector_name,
            change_points=list(detection.change_points),
            params=dict(detection.params),
            scores=list(detection.scores),
            elapsed_ms=detection.elapsed_ms,
            evaluation_report=report,
            dataset_summary=data.summary(),
        )
        self._history.append(result)
        return result

    def run_synthetic(
        self,
        kind: str = "constant",
        n_samples: int = 1000,
        n_change_points: int = 3,
        detector: str | None = None,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> PipelineResult:
        """生成合成数据并直接跑一次端到端。"""
        dataset = generate_dataset(
            kind=kind,
            n_samples=n_samples,
            n_change_points=n_change_points,
            seed=self._config.seed if seed is None else seed,
        )
        return self.run(dataset=dataset, detector=detector, params=params)

    def run_many(
        self,
        datasets: Sequence[Dataset],
        detector: str,
        params: dict[str, Any] | None = None,
    ) -> list[PipelineResult]:
        """在多个数据集上跑同一个检测器。"""
        return [self.run(dataset=data, detector=detector, params=params) for data in datasets]

    def tune(
        self,
        dataset: Dataset,
        detector: str,
        search_space_dict: dict[str, Any],
        metric: str | None = None,
        n_trials: int | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """用 Optuna 为指定检测器调参（hpo 层延迟导入）。

        Args:
            dataset: 带真值的数据集。
            detector: 检测器名。
            search_space_dict: ``{参数名: {...}}``，见 ``hpo.make_search_space``。
            metric: 优化指标（EvalReport 字段名），默认取 config.hpo_metric。
            n_trials / timeout_s: 覆盖 config。

        Returns:
            ``hpo.HPOResult``。
        """
        # 延迟导入：optuna 较重，且保持 pipeline -> hpo 的单向依赖
        from ..hpo import make_search_space, run_hpo  # noqa: PLC0415

        data = validate_dataset(dataset, min_size=self._config.min_size)
        if not data.change_points:
            raise PipelineError("调参需要带真值变化点的数据集", dataset=data.name)
        space = make_search_space(search_space_dict)
        metric_name = metric or self._config.hpo_metric
        tolerance = self._config.tolerance

        def objective(params: dict[str, Any], trial: Any) -> float:
            result = self.run(
                dataset=data,
                detector=detector,
                params=params,
                tolerance=tolerance,
            )
            if result.evaluation_report is None:
                return float("nan")
            value = getattr(result.evaluation_report, metric_name, None)
            if value is None:
                raise PipelineError("未知优化指标", metric=metric_name)
            trial.set_user_attr("f1", result.evaluation_report.f1)
            return float(value)

        return run_hpo(
            objective,
            space,
            config=self._config,
            n_trials=n_trials,
            timeout_s=timeout_s,
        )

    @staticmethod
    def _coerce_points(points: Any, n_samples: int) -> list[int]:
        """把检测器输出归一化为合法、升序、去重的变化点列表。"""
        if points is None:
            return []
        raw = (
            list(np.asarray(points).reshape(-1).tolist())
            if isinstance(points, np.ndarray)
            else list(points)
        )
        cleaned: list[int] = []
        for item in raw:
            value = int(round(float(item)))
            if 1 <= value <= n_samples - 1:
                cleaned.append(value)
            else:
                warnings.warn(
                    f"丢弃越界变化点 {value}（合法范围 1~{n_samples - 1}）",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return sorted(set(cleaned))
