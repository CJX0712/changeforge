"""Optuna 超参搜索基建。

设计要点：
    1. 目标函数由调用方注入，签名 ``objective(params: dict, trial) -> float``，
       这样 hpo 层不必知道检测器与指标细节（保持单向无环）；
    2. ``n_jobs`` 恒为 1 —— Windows 下 joblib/多进程并发必崩；
    3. trial 内抛出的业务异常可配置为 "raise"（严格）或 "skip"（跳过该试验）。

作者: 晨星
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import optuna
from optuna.trial import Trial

from ..core.config import Config
from ..core.errors import InvalidSearchSpaceError, StudyFailedError

Objective = Callable[[dict[str, Any], Trial], float]

PARAM_KINDS = ("int", "float", "loguniform", "categorical")

__all__ = [
    "HPOResult",
    "Objective",
    "SearchSpace",
    "make_search_space",
    "run_hpo",
]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """单个超参数的搜索定义。

    kind:
        int          -> trial.suggest_int(name, low, high, step)
        float        -> trial.suggest_float(name, low, high, step)
        loguniform   -> trial.suggest_float(name, low, high, log=True)
        categorical  -> trial.suggest_categorical(name, choices)
    """

    kind: str
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] = ()
    default: Any = None

    def validate(self, name: str) -> None:
        if self.kind not in PARAM_KINDS:
            raise InvalidSearchSpaceError(
                "不支持的参数类型", param=name, kind=self.kind, supported=list(PARAM_KINDS)
            )
        if self.kind == "categorical":
            if not self.choices:
                raise InvalidSearchSpaceError("categorical 必须给出 choices", param=name)
            return
        if self.low is None or self.high is None:
            raise InvalidSearchSpaceError("需要 low 与 high", param=name)
        if float(self.high) <= float(self.low):
            raise InvalidSearchSpaceError("high 必须大于 low", param=name)
        if self.kind == "loguniform" and float(self.low) <= 0.0:
            raise InvalidSearchSpaceError("loguniform 的 low 必须 > 0", param=name)

    def suggest(self, trial: Trial, name: str) -> Any:
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            return trial.suggest_int(name, int(self.low), int(self.high), step=self.step or 1)
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            return trial.suggest_float(name, float(self.low), float(self.high), step=self.step)
        if self.kind == "loguniform":
            assert self.low is not None and self.high is not None
            return trial.suggest_float(
                name, float(self.low), float(self.high), log=True, step=self.step
            )
        return trial.suggest_categorical(name, list(self.choices))


@dataclass(slots=True, eq=False)
class SearchSpace:
    """超参搜索空间：``{参数名: ParamSpec}``。"""

    specs: dict[str, ParamSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, spec in self.specs.items():
            if not isinstance(spec, ParamSpec):
                raise InvalidSearchSpaceError("spec 必须是 ParamSpec 实例", param=name)
            spec.validate(name)

    def names(self) -> list[str]:
        return sorted(self.specs)

    def defaults(self) -> dict[str, Any]:
        return {
            name: spec.default for name, spec in self.specs.items() if spec.default is not None
        }

    def suggest(self, trial: Trial) -> dict[str, Any]:
        return {name: self.specs[name].suggest(trial, name) for name in self.names()}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, spec in self.specs.items():
            payload[name] = {
                "kind": spec.kind,
                "low": spec.low,
                "high": spec.high,
                "step": spec.step,
                "choices": list(spec.choices),
                "default": spec.default,
            }
        return payload


def make_search_space(specs: Mapping[str, Any]) -> SearchSpace:
    """从 ``{name: dict}`` 便捷构造 SearchSpace（dict 会转成 ParamSpec）。"""
    converted: dict[str, ParamSpec] = {}
    for name, raw in specs.items():
        if isinstance(raw, ParamSpec):
            converted[name] = raw
        elif isinstance(raw, Mapping):
            unknown = sorted(set(raw) - {"kind", "low", "high", "step", "choices", "default"})
            if unknown:
                raise InvalidSearchSpaceError("未知搜索空间字段", param=name, keys=unknown)
            converted[name] = ParamSpec(
                kind=str(raw.get("kind", "float")),
                low=raw.get("low"),
                high=raw.get("high"),
                step=raw.get("step"),
                choices=tuple(raw.get("choices", ())),
                default=raw.get("default"),
            )
        else:
            raise InvalidSearchSpaceError("搜索空间条目必须是 dict 或 ParamSpec", param=name)
    return SearchSpace(converted)


@dataclass(slots=True, eq=False)
class HPOResult:
    """一次超参搜索的结果。"""

    study_name: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    n_completed: int
    direction: str
    elapsed_ms: float
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "best_params": dict(self.best_params),
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "direction": self.direction,
            "elapsed_ms": self.elapsed_ms,
            "trials": [dict(trial) for trial in self.trials],
        }


def _worst_value(direction: str) -> float:
    return float("-inf") if direction == "maximize" else float("inf")


def run_hpo(
    objective: Objective,
    search_space: SearchSpace,
    config: Config | None = None,
    direction: Literal["maximize", "minimize"] = "maximize",
    study_name: str | None = None,
    n_trials: int | None = None,
    timeout_s: float | None = None,
    on_error: Literal["raise", "skip"] | None = None,
) -> HPOResult:
    """执行 Optuna 搜索。

    Args:
        objective: ``(params, trial) -> float``，可抛 ``optuna.TrialPruned`` 剪枝。
        search_space: 参数空间。
        config: 提供 seed / n_trials / timeout / strict 默认值。
        direction: 优化方向。
        study_name: study 名，None 时自动生成。
        n_trials: 覆盖 config.hpo_n_trials。
        timeout_s: 覆盖 config.hpo_timeout_s。
        on_error: trial 抛异常时的处理；None 时取 config.strict 推导。

    Returns:
        HPOResult。

    Raises:
        StudyFailedError: 严格模式下某个 trial 抛异常，或 study 无成功试验。
    """
    config = config or Config()
    if not isinstance(search_space, SearchSpace):
        raise InvalidSearchSpaceError("search_space 必须是 SearchSpace 实例")
    if not search_space.specs:
        raise InvalidSearchSpaceError("搜索空间为空")

    trials = n_trials if n_trials is not None else config.hpo_n_trials
    timeout = timeout_s if timeout_s is not None else config.hpo_timeout_s
    error_policy = on_error or ("raise" if config.strict else "skip")
    worst = _worst_value(direction)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(config.seed))
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name or f"changeforge-{int(time.time())}",
    )

    history: list[dict[str, Any]] = []

    def wrapped(trial: Trial) -> float:
        params = search_space.suggest(trial)
        try:
            value = float(objective(params, trial))
        except optuna.TrialPruned:
            raise
        except optuna.exceptions.OptunaError:
            raise
        except Exception as exc:  # noqa: BLE001 - 业务异常统一转换为策略行为
            if error_policy == "raise":
                raise StudyFailedError(
                    "试验执行失败", trial=trial.number, params=params, reason=repr(exc)
                ) from exc
            history.append({"number": trial.number, "params": params, "state": "fail"})
            return worst
        history.append(
            {"number": trial.number, "params": params, "value": value, "state": "ok"}
        )
        return value

    start = time.perf_counter()
    # Windows 并发坑：n_jobs 恒为 1
    study.optimize(wrapped, n_trials=int(trials), timeout=timeout, n_jobs=1)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise StudyFailedError(
            "study 没有成功完成的试验", study=study.study_name, n_trials=len(study.trials)
        )
    return HPOResult(
        study_name=study.study_name,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        n_completed=len(completed),
        direction=direction,
        elapsed_ms=elapsed_ms,
        trials=history,
    )
