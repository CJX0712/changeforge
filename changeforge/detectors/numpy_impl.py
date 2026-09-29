"""纯 numpy 检测器实现（离线兜底，零第三方依赖即可运行）。

注册名:
    numpy-pelt-l2     PELT + l2 代价（beta = lam * sigma_hat^2 * log n）
    numpy-binseg-l2   二分分割
    numpy-window-l2   滑动窗口
    numpy-dynp-l2     精确 DP（oracle-K 对照用）
    numpy-cusum       在线双侧 CUSUM
    numpy-slidingt    滑动 Welch-t + BH 校正
    numpy-mmd-rff     RFF 近似核 CPD（PELT on 随机傅里叶特征）

作者: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.algorithms import (
    beta_from_lambda,
    binseg_l2,
    cusum_online,
    dynp_l2,
    pelt_l2,
    rff_features,
    sliding_welch_t,
    window_l2,
)
from ..core.errors import InvalidDetectorParamsError
from ..core.types import Signal
from . import register

__all__ = [
    "NumpyPeltL2",
    "NumpyBinsegL2",
    "NumpyWindowL2",
    "NumpyDynpL2",
    "NumpyCusum",
    "NumpySlidingT",
    "NumpyMmdRff",
]


class _BaseNumpyDetector:
    """Detector 协议公共骨架（name/fit/predict/get_params/set_params）。"""

    name: str = "numpy-base"
    _allowed: tuple[str, ...] = ()

    def __init__(self, **params: Any) -> None:
        self._params: dict[str, Any] = {}
        self._signal: np.ndarray | None = None
        if params:
            self.set_params(**params)

    # -- 协议方法 -----------------------------------------------------------
    def fit(self, signal: Signal) -> _BaseNumpyDetector:
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim not in (1, 2) or arr.shape[0] < 3:
            raise InvalidDetectorParamsError("signal 至少需要 3 个样本", shape=arr.shape)
        self._signal = arr
        return self

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any) -> _BaseNumpyDetector:
        for key in params:
            if key not in self._allowed:
                raise InvalidDetectorParamsError(
                    "未知参数", detector=self.name, param=key, allowed=list(self._allowed)
                )
        self._params.update(params)
        return self

    # -- 内部工具 -----------------------------------------------------------
    def _min_size(self, n: int) -> int:
        return max(2, int(self._params.get("min_size", max(2, round(0.01 * n)))))

    def _1d(self) -> np.ndarray:
        assert self._signal is not None
        return self._signal.reshape(-1) if self._signal.ndim > 1 else self._signal

    def _2d(self) -> np.ndarray:
        assert self._signal is not None
        return self._signal.reshape(-1, 1) if self._signal.ndim == 1 else self._signal


class NumpyPeltL2(_BaseNumpyDetector):
    name = "numpy-pelt-l2"
    _allowed = ("lam", "min_size")

    def predict(self) -> list[int]:
        assert self._signal is not None
        n = int(self._signal.shape[0])
        lam = float(self._params.get("lam", 3.0))
        beta = beta_from_lambda(self._signal, lam)
        return [int(c) for c in pelt_l2(self._signal, beta, self._min_size(n))]


class NumpyBinsegL2(_BaseNumpyDetector):
    name = "numpy-binseg-l2"
    _allowed = ("lam", "min_size", "n_bkps")

    def predict(self) -> list[int]:
        assert self._signal is not None
        n = int(self._signal.shape[0])
        ms = self._min_size(n)
        n_bkps = self._params.get("n_bkps")
        if n_bkps is not None:
            cps, _ = binseg_l2(self._signal, n_bkps=int(n_bkps), min_size=ms)
        else:
            lam = float(self._params.get("lam", 3.0))
            pen = beta_from_lambda(self._signal, lam)
            cps, _ = binseg_l2(self._signal, pen=pen, min_size=ms)
        return [int(c) for c in cps]


class NumpyWindowL2(_BaseNumpyDetector):
    name = "numpy-window-l2"
    _allowed = ("width", "min_size", "n_bkps", "lam")

    def predict(self) -> list[int]:
        assert self._signal is not None
        n = int(self._signal.shape[0])
        ms = self._min_size(n)
        width = int(self._params.get("width", max(10, 4 * ms)))
        n_bkps = self._params.get("n_bkps")
        pen = None
        if n_bkps is None:
            lam = float(self._params.get("lam", 3.0))
            pen = beta_from_lambda(self._signal, lam)
        cps = window_l2(
            self._signal,
            width=width,
            min_size=ms,
            n_bkps=None if n_bkps is None else int(n_bkps),
            pen=pen,
        )
        return [int(c) for c in cps]


class NumpyDynpL2(_BaseNumpyDetector):
    name = "numpy-dynp-l2"
    _allowed = ("n_bkps", "min_size")

    def predict(self) -> list[int]:
        assert self._signal is not None
        n = int(self._signal.shape[0])
        n_bkps = int(self._params.get("n_bkps", 1))
        return [int(c) for c in dynp_l2(self._signal, n_bkps, self._min_size(n))]


class NumpyCusum(_BaseNumpyDetector):
    name = "numpy-cusum"
    _allowed = ("delta", "h", "burn", "cooldown")

    def predict(self) -> list[int]:
        kw = {k: v for k, v in self._params.items() if k in ("delta", "h", "burn", "cooldown")}
        return [int(c) for c in cusum_online(self._1d(), **kw)]


class NumpySlidingT(_BaseNumpyDetector):
    name = "numpy-slidingt"
    _allowed = ("window", "alpha", "min_size")

    def predict(self) -> list[int]:
        n = int(self._signal.shape[0]) if self._signal is not None else 0
        kw: dict[str, Any] = {}
        if "window" in self._params:
            kw["window"] = int(self._params["window"])
        if "alpha" in self._params:
            kw["alpha"] = float(self._params["alpha"])
        kw["min_size"] = self._min_size(n)
        return [int(c) for c in sliding_welch_t(self._1d(), **kw)]


class NumpyMmdRff(_BaseNumpyDetector):
    name = "numpy-mmd-rff"
    _allowed = ("lam", "min_size", "n_features", "seed")

    def predict(self) -> list[int]:
        assert self._signal is not None
        n = int(self._signal.shape[0])
        lam = float(self._params.get("lam", 3.0))
        ms = self._min_size(n)
        z = rff_features(
            self._signal,
            n_features=int(self._params.get("n_features", 256)),
            seed=int(self._params.get("seed", 0)),
        )
        beta = beta_from_lambda(z, lam)
        return [int(c) for c in pelt_l2(z, beta, ms)]


# --------------------------------------------------------------------------- 注册
@register("numpy-pelt-l2", doc="PELT + l2，纯 numpy 兜底（beta=lam*sigma^2*log n）")
def _build_pelt(**params: Any) -> NumpyPeltL2:
    return NumpyPeltL2(**params)


@register("numpy-binseg-l2", doc="二分分割 BinSeg + l2，纯 numpy 兜底")
def _build_binseg(**params: Any) -> NumpyBinsegL2:
    return NumpyBinsegL2(**params)


@register("numpy-window-l2", doc="滑动窗口 + l2，纯 numpy 兜底")
def _build_window(**params: Any) -> NumpyWindowL2:
    return NumpyWindowL2(**params)


@register("numpy-dynp-l2", doc="精确 DP（oracle-K 专用，禁止进主对比表）")
def _build_dynp(**params: Any) -> NumpyDynpL2:
    return NumpyDynpL2(**params)


@register("numpy-cusum", doc="在线双侧 CUSUM（Page 检验）")
def _build_cusum(**params: Any) -> NumpyCusum:
    return NumpyCusum(**params)


@register("numpy-slidingt", doc="滑动 Welch-t + BH 多重检验校正")
def _build_slidingt(**params: Any) -> NumpySlidingT:
    return NumpySlidingT(**params)


@register("numpy-mmd-rff", doc="RFF 近似核 CPD（O(nD)，KernelCPD 的轻量替代）")
def _build_mmdrff(**params: Any) -> NumpyMmdRff:
    return NumpyMmdRff(**params)
