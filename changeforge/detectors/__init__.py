"""变化点检测器注册表（**占位模块**）。

状态：施工中。算法实现（PECUC / RuLSIF / BOCPD / KL-CPD / ruptures 基线 等）
待算法科学家方案确定后在 ``detectors/`` 下逐模块落地，这里只提供注册与
解析机制，保证上层 pipeline / CLI / HPO 现在就能跑通全链路。

约定：
    1. 每个检测器用 ``@register("name")`` 装饰工厂函数或直接注册类；
    2. 工厂/类需满足 ``core.interfaces.Detector`` 协议（name / fit / predict /
       get_params / set_params）；
    3. :func:`build` 负责按名构造并把超参数传给 ``set_params``，未知参数由
       实现方抛出 ``InvalidDetectorParamsError``；
    4. 不要把内部提示参数（如 verbose、n_jobs 提示位）塞进 ``**params``
       传给估计器，避免参数泄漏。

作者: 晨星
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.errors import DetectorNotFoundError, InvalidDetectorParamsError
from ..core.interfaces import Detector
from ..core.types import Signal

Factory = Callable[..., Detector]

_REGISTRY: dict[str, Factory] = {}
_DOCSTRINGS: dict[str, str] = {}

__all__ = [
    "build",
    "get_registry_doc",
    "is_registered",
    "list_detectors",
    "register",
]


def register(name: str, doc: str | None = None) -> Callable[[Factory], Factory]:
    """注册检测器工厂。

    Args:
        name: registry key，CLI / 配置里的 ``detector`` 字段即该名字。
        doc: 一行说明，供 ``list-detectors`` 展示。

    Example:
        >>> @register("my-detector", doc="示例检测器")
        ... def build_my_detector(**params):
        ...     ...
    """

    def decorator(factory: Factory) -> Factory:
        key = str(name).strip()
        if not key:
            raise InvalidDetectorParamsError("检测器名不能为空")
        _REGISTRY[key] = factory
        _DOCSTRINGS[key] = (
            doc or (factory.__doc__ or "").strip().splitlines()[0] if (factory.__doc__) else ""
        )
        return factory

    return decorator


def list_detectors() -> list[str]:
    """返回已注册检测器名（升序）。"""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def get_registry_doc(name: str) -> str:
    return _DOCSTRINGS.get(name, "")


def build(name: str, **params: Any) -> Detector:
    """按名构造检测器并套用超参数。

    Raises:
        DetectorNotFoundError: 名字未注册（含 registry 为空的情况）。
        InvalidDetectorParamsError: 超参数不被实现方接受。
    """
    key = str(name).strip()
    if key not in _REGISTRY:
        raise DetectorNotFoundError(
            "未找到检测器（算法模块尚未接入时可先用占位注册表自检）",
            detector=key,
            available=list_detectors(),
        )
    detector = _REGISTRY[key](**params)
    if not hasattr(detector, "predict") or not hasattr(detector, "fit"):
        raise InvalidDetectorParamsError(
            "检测器未实现 fit/predict", detector=key, type=type(detector).__name__
        )
    return detector


def predict_with(name: str, signal: Signal, **params: Any) -> list[int]:
    """便捷入口：构造 -> fit -> predict（供脚本与演示使用）。"""
    detector = build(name, **params)
    detector.fit(signal)
    return [int(c) for c in detector.predict()]
