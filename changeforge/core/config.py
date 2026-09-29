"""配置对象与 ``ENV_CHANGEFORGE_*`` 环境变量覆盖。

优先级（低 -> 高）::

    dataclass 默认值  <  ENV_CHANGEFORGE_* 环境变量  <  显式 override 参数

环境变量名规则：``ENV_CHANGEFORGE_`` + 字段名大写，例如 ``ENV_CHANGEFORGE_SEED=7``。
取值先按 JSON 解析，失败再按目标类型强制转换；无法转换时抛 :class:`InvalidConfigValue`。

作者: 晨星
"""

from __future__ import annotations

import json
import os
import sys
import types
import typing
import warnings
from dataclasses import dataclass, fields, replace

from .errors import InvalidConfigValue, UnknownConfigKey

ENV_PREFIX = "ENV_CHANGEFORGE_"

_TRUE_LITERALS = frozenset({"1", "true", "yes", "on"})
_FALSE_LITERALS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class Config:
    """ChangeForge 全局配置。

    Attributes:
        seed: 随机种子（合成数据 / HPO 采样器共用）。
        n_jobs: 并行度。Windows 下 joblib/sklearn 并发易崩，非 1 一律被钳到 1。
        log_level: 日志级别名，如 DEBUG/INFO/WARNING。
        tolerance: 评测时变化点匹配的容差（样本点数）。
        min_size: 检测器允许的最小段长度。
        detector: 默认检测器名（registry 中的 key）。
        hpo_n_trials: Optuna 试验次数。
        hpo_timeout_s: Optuna 超时秒数；None 表示不限时。
        hpo_metric: HPO 优化目标指标名（eval/metrics.py 中 EvalReport 的字段）。
        output_dir: 产物（benchmark.json 等）输出目录。
        strict: True 时把可恢复的异常升级为报错，False 时降级为告警。
    """

    seed: int = 42
    n_jobs: int = 1
    log_level: str = "INFO"
    tolerance: int = 5
    min_size: int = 5
    detector: str = "auto"
    hpo_n_trials: int = 20
    hpo_timeout_s: float | None = None
    hpo_metric: str = "f1"
    output_dir: str = "artifacts"
    strict: bool = True

    def __post_init__(self) -> None:
        # Windows 并发坑：joblib/sklearn 多进程在 Windows 上极易崩溃，强制单进程。
        if self.n_jobs != 1 and sys.platform.startswith("win"):
            object.__setattr__(self, "n_jobs", 1)
            warnings.warn(
                "Windows 平台不支持 n_jobs != 1，已自动钳制为 1（避免 joblib 崩溃）",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.tolerance < 0:
            raise InvalidConfigValue("tolerance 不能为负数", tolerance=self.tolerance)
        if self.min_size < 1:
            raise InvalidConfigValue("min_size 必须 >= 1", min_size=self.min_size)
        if self.hpo_n_trials < 1:
            raise InvalidConfigValue("hpo_n_trials 必须 >= 1", hpo_n_trials=self.hpo_n_trials)

    def with_overrides(self, **overrides: object) -> Config:
        """返回带覆盖值的新 Config（原对象不变）。未知键抛 UnknownConfigKey。"""
        cleaned = {k: v for k, v in overrides.items() if v is not None}
        unknown = sorted(set(cleaned) - {f.name for f in fields(Config)})
        if unknown:
            raise UnknownConfigKey("收到未知配置项", keys=unknown)
        return replace(self, **cleaned) if cleaned else self

    def to_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(Config)}

    def env_var(self, field_name: str) -> str:
        return f"{ENV_PREFIX}{field_name.upper()}"


def _unwrap_optional(annotation: object) -> tuple[object, bool]:
    """把 ``float | None`` / ``Optional[float]`` 拆成 (内层类型, 是否可选)。"""
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _coerce_scalar(raw: object, annotation: object, field_name: str, source: str) -> object:
    """把原始值强制转换为字段注解类型，失败抛 InvalidConfigValue。"""
    inner, optional = _unwrap_optional(annotation)
    if raw is None:
        if optional:
            return None
        raise InvalidConfigValue("配置项不接受 None", key=field_name, source=source)
    if inner is bool:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE_LITERALS:
            return True
        if text in _FALSE_LITERALS:
            return False
        raise InvalidConfigValue("布尔解析失败", key=field_name, value=raw, source=source)
    if inner is int:
        try:
            if isinstance(raw, bool):
                raise TypeError("bool 不是合法 int 配置值")
            return int(str(raw).strip()) if isinstance(raw, str) else int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise InvalidConfigValue(
                "整数解析失败", key=field_name, value=raw, source=source
            ) from exc
    if inner is float:
        try:
            return float(str(raw).strip()) if isinstance(raw, str) else float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise InvalidConfigValue(
                "浮点数解析失败", key=field_name, value=raw, source=source
            ) from exc
    if inner is str:
        return str(raw)
    return raw


def _type_hints() -> dict[str, object]:
    return typing.get_type_hints(Config)


def from_env(env: dict[str, str] | None = None, **overrides: object) -> Config:
    """从环境变量构建 Config，再叠加显式覆盖。"""
    source = os.environ if env is None else env
    hints = _type_hints()
    values: dict[str, object] = {}
    for name, hint in hints.items():
        raw = source.get(f"{ENV_PREFIX}{name.upper()}")
        if raw is None:
            continue
        if name in {"seed", "tolerance", "min_size", "n_jobs", "hpo_n_trials"} or isinstance(
            raw, (int, float, bool)
        ):
            values[name] = _coerce_scalar(raw, hint, name, "env")
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            values[name] = _coerce_scalar(raw, hint, name, "env")
        else:
            values[name] = _coerce_scalar(parsed, hint, name, "env")
    base = Config(**values)  # type: ignore[arg-type]
    return base.with_overrides(**overrides)


def load_config(**overrides: object) -> Config:
    """加载配置：默认值 -> 环境变量 -> 显式覆盖。"""
    return from_env(**overrides)


def config_from_file(path: str) -> Config:
    """从 JSON 文件加载部分配置（作为 overrides 使用）。

    未出现的键保持默认/环境变量值；未知键抛 UnknownConfigKey。
    """
    with open(path, encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise InvalidConfigValue("配置文件不是合法 JSON", path=path) from exc
    if not isinstance(payload, dict):
        raise InvalidConfigValue("配置文件顶层必须是对象", path=path)
    return load_config(**payload)
