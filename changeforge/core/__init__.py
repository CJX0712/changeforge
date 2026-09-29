"""ChangeForge 核心层：类型 / 错误码 / 配置 / 协议 / 校验。

本层只依赖标准库与 numpy，禁止反向依赖 data、detectors、hpo、eval、pipeline。
"""

from __future__ import annotations

from .config import ENV_PREFIX, Config, load_config
from .errors import (
    ERROR_CLASSES,
    BenchmarkError,
    ChangeForgeError,
    CLIError,
    ConfigError,
    DataError,
    DatasetNotFoundError,
    DetectorError,
    DetectorNotFoundError,
    DetectorNotFittedError,
    EvaluationError,
    EnvOverrideError,
    HPOError,
    InvalidChangePointsError,
    InvalidConfigValue,
    InvalidDetectorParamsError,
    InvalidSearchSpaceError,
    InvalidSignalError,
    PipelineError,
    StudyFailedError,
    UnknownConfigKey,
    UnsupportedDataFormatError,
    get_error_class,
)
from .interfaces import Detector, SupportsFitPredict
from .types import (
    ChangePoint,
    Dataset,
    DetectionResult,
    EvalReport,
    Signal,
)
from .validate import (
    validate_change_points,
    validate_dataset,
    validate_signal,
)

__all__ = [
    "ENV_PREFIX",
    "BenchmarkError",
    "CLIError",
    "ChangeForgeError",
    "ChangePoint",
    "Config",
    "ConfigError",
    "ERROR_CLASSES",
    "DataError",
    "Dataset",
    "DatasetNotFoundError",
    "DetectionResult",
    "Detector",
    "DetectorError",
    "DetectorNotFoundError",
    "DetectorNotFittedError",
    "EnvOverrideError",
    "EvalReport",
    "EvaluationError",
    "HPOError",
    "InvalidChangePointsError",
    "InvalidConfigValue",
    "InvalidDetectorParamsError",
    "InvalidSearchSpaceError",
    "InvalidSignalError",
    "PipelineError",
    "Signal",
    "StudyFailedError",
    "SupportsFitPredict",
    "UnknownConfigKey",
    "UnsupportedDataFormatError",
    "get_error_class",
    "load_config",
    "validate_change_points",
    "validate_dataset",
    "validate_signal",
]
