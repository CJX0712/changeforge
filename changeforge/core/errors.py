"""ChangeForge 统一错误码体系（E100 ~ E500）。

分段:

    E1xx  配置与环境变量
    E2xx  数据
    E3xx  检测器
    E4xx  超参搜索（HPO）
    E5xx  流水线 / 评测 / 基准 / CLI

所有异常都带 ``code`` 与结构化 ``context``，便于 CLI 与日志统一呈现。

作者: 晨星
"""

from __future__ import annotations

from typing import Any


class ChangeForgeError(Exception):
    """ChangeForge 所有异常的基类（E000）。"""

    code: str = "E000"
    default_message: str = "ChangeForge 未分类错误"

    def __init__(self, message: str | None = None, **context: Any) -> None:
        self.message = message if message else self.default_message
        self.context: dict[str, Any] = dict(context)
        super().__init__(self.render())

    def render(self) -> str:
        head = f"[{self.code}] {self.message}"
        if self.context:
            extra = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
            head = f"{head} ({extra})"
        return head

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


# --------------------------------------------------------------------------- E1xx 配置
class ConfigError(ChangeForgeError):
    code = "E100"
    default_message = "配置错误"


class InvalidConfigValue(ConfigError):
    code = "E101"
    default_message = "配置项取值非法"


class UnknownConfigKey(ConfigError):
    code = "E102"
    default_message = "未知配置项"


class EnvOverrideError(ConfigError):
    code = "E103"
    default_message = "环境变量覆盖失败"


# --------------------------------------------------------------------------- E2xx 数据
class DataError(ChangeForgeError):
    code = "E200"
    default_message = "数据错误"


class InvalidSignalError(DataError):
    code = "E201"
    default_message = "信号数组非法"


class InvalidChangePointsError(DataError):
    code = "E202"
    default_message = "变化点索引非法"


class DatasetNotFoundError(DataError):
    code = "E203"
    default_message = "数据集不存在"


class UnsupportedDataFormatError(DataError):
    code = "E204"
    default_message = "不支持的数据格式"


# --------------------------------------------------------------------------- E3xx 检测器
class DetectorError(ChangeForgeError):
    code = "E300"
    default_message = "检测器错误"


class DetectorNotFoundError(DetectorError):
    code = "E301"
    default_message = "未找到指定检测器"


class DetectorNotFittedError(DetectorError):
    code = "E302"
    default_message = "检测器尚未 fit"


class InvalidDetectorParamsError(DetectorError):
    code = "E303"
    default_message = "检测器超参数非法"


# --------------------------------------------------------------------------- E4xx HPO
class HPOError(ChangeForgeError):
    code = "E400"
    default_message = "超参搜索错误"


class StudyFailedError(HPOError):
    code = "E401"
    default_message = "Optuna study 执行失败"


class InvalidSearchSpaceError(HPOError):
    code = "E402"
    default_message = "搜索空间定义非法"


# --------------------------------------------------------------------------- E5xx 流水线
class PipelineError(ChangeForgeError):
    code = "E500"
    default_message = "流水线错误"


class EvaluationError(PipelineError):
    code = "E501"
    default_message = "评测失败"


class BenchmarkError(PipelineError):
    code = "E502"
    default_message = "基准测试失败"


class CLIError(PipelineError):
    code = "E503"
    default_message = "命令行参数错误"


def _build_registry() -> dict[str, type[ChangeForgeError]]:
    classes: list[type[ChangeForgeError]] = [
        ChangeForgeError,
        ConfigError,
        InvalidConfigValue,
        UnknownConfigKey,
        EnvOverrideError,
        DataError,
        InvalidSignalError,
        InvalidChangePointsError,
        DatasetNotFoundError,
        UnsupportedDataFormatError,
        DetectorError,
        DetectorNotFoundError,
        DetectorNotFittedError,
        InvalidDetectorParamsError,
        HPOError,
        StudyFailedError,
        InvalidSearchSpaceError,
        PipelineError,
        EvaluationError,
        BenchmarkError,
        CLIError,
    ]
    return {cls.code: cls for cls in classes}


ERROR_CLASSES: dict[str, type[ChangeForgeError]] = _build_registry()


def get_error_class(code: str) -> type[ChangeForgeError]:
    """按错误码取异常类；未知码回落到基类。"""
    return ERROR_CLASSES.get(code.upper(), ChangeForgeError)
