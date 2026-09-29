"""pytest 公共夹具。

``pyproject.toml`` 已配置 ``pythonpath = ["."]``，这里再补一道 sys.path 保险，
保证无论从哪个目录执行 pytest 都能 import 到 changeforge 包。

作者: 晨星
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class DummyDetector:
    """测试用假检测器：返回构造时给定的固定变化点。"""

    name = "dummy"

    def __init__(self, change_points=None, **params):
        self._change_points = list(change_points or [])
        self._params = dict(params)
        self._signal = None

    def fit(self, signal: np.ndarray):
        self._signal = signal
        return self

    def predict(self) -> list[int]:
        if self._signal is None:
            raise RuntimeError("not fitted")
        return [int(c) for c in self._change_points]

    def get_params(self) -> dict:
        return {"change_points": list(self._change_points), **dict(self._params)}

    def set_params(self, **params):
        self._params.update(params)
        return self


@pytest.fixture
def dummy_detector_module():
    """注册一个 dummy 检测器，用例结束后自动摘除（避免污染其他测试）。"""
    from changeforge import detectors

    def factory(**params):
        return DummyDetector(**params)

    detectors._REGISTRY["dummy"] = factory
    detectors._DOCSTRINGS["dummy"] = "测试用假检测器"
    yield detectors
    detectors._REGISTRY.pop("dummy", None)
    detectors._DOCSTRINGS.pop("dummy", None)


@pytest.fixture
def empty_registry(monkeypatch):
    """把检测器注册表清空，用于验证 DetectorNotFoundError 路径。"""
    from changeforge import detectors

    monkeypatch.setattr(detectors, "_REGISTRY", {})
    monkeypatch.setattr(detectors, "_DOCSTRINGS", {})
    return detectors


@pytest.fixture
def tiny_signal():
    """100 点、2 个真值变化点（30 / 70）的分段常数信号。"""
    rng = np.random.default_rng(0)
    signal = np.concatenate(
        [
            rng.normal(0.0, 0.1, size=30),
            rng.normal(3.0, 0.1, size=40),
            rng.normal(-2.0, 0.1, size=30),
        ]
    )
    return signal, [30, 70]
