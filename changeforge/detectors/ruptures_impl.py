"""ruptures 后端 adapter（SOTA 主后端，惰性导入 + 能力探测 + 自动降级）。

实测事实（2026-09-29, Windows 11 / Python 3.13.12 / ruptures 1.1.10）:
    - 类名是 ``rpt.Binseg``（小写 s）、``rpt.Window``、``rpt.Dynp``、
      ``rpt.KernelCPD``、``rpt.Pelt``；
    - Pelt 支持 8 种 model: l1/l2/normal/rbf/linear/ar/rank/clinear；
    - 返回 dtype 不一致（int / np.int64 / np.int32），输出必须 int() 显式转换；
    - ``predict()`` 返回的列表末尾含 n，比较前必须去掉；
    - 本机无 .pyd/.so 扩展 => ruptures 走纯 Python 路径，无 Cython 加速；
    - ``normal`` cost 会抛 UserWarning（v1.1.5 起给协方差加 bias）。

作者: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import InvalidDetectorParamsError
from ..core.types import Signal
from . import register

__all__ = ["ruptures_available", "RupturesDetector", "RUPTURES_MIN_VERSION"]

RUPTURES_MIN_VERSION = (1, 1, 9)
_KERNEL_N_MAX = 5000  # KernelCPD O(n^2) 内存硬上限（n=20000 -> ~3.2GB OOM）


def ruptures_available() -> bool:
    """能力探测：import 成功且版本 >= 1.1.9 即可用。"""
    try:
        import ruptures

        parts = str(getattr(ruptures, "__version__", "0")).lstrip("v").split(".")
        version = tuple(int(p) for p in parts[:3])
        return version >= RUPTURES_MIN_VERSION
    except Exception:
        return False


def _backend_version() -> str:
    import ruptures

    return str(getattr(ruptures, "__version__", "unknown"))


class RupturesDetector:
    """ruptures 检测器适配层。

    params:
        algo: pelt | binseg | window | dynp | kernel
        model: l1|l2|normal|rbf|linear|ar|rank|clinear（kernel 用 kernel 参数）
        lam: 归一化惩罚系数（beta = lam * sigma_hat^2 * log n）
        min_size: 最小段长
        n_bkps: dynp/binseg/window 的 K（oracle-K 模式；pelt 用 pen）
        kernel: KernelCPD 的核（默认 rbf）
    """

    name = "ruptures"
    _allowed = ("algo", "model", "lam", "min_size", "n_bkps", "kernel")

    def __init__(self, algo: str = "pelt", **params: Any) -> None:
        self._params: dict[str, Any] = {"algo": algo}
        self._params.update(params)
        self._signal: np.ndarray | None = None
        unknown = [k for k in self._params if k not in self._allowed]
        if unknown:
            raise InvalidDetectorParamsError(
                "未知参数", detector=self.name, param=unknown[0], allowed=list(self._allowed)
            )

    # -- 协议 ---------------------------------------------------------------
    def fit(self, signal: Signal) -> RupturesDetector:
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] < 3:
            raise InvalidDetectorParamsError("signal 至少需要 3 个样本", shape=arr.shape)
        self._signal = arr
        return self

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any) -> RupturesDetector:
        for key in params:
            if key not in self._allowed:
                raise InvalidDetectorParamsError(
                    "未知参数", detector=self.name, param=key, allowed=list(self._allowed)
                )
        self._params.update(params)
        return self

    def predict(self) -> list[int]:
        assert self._signal is not None
        import ruptures as rpt  # 惰性导入：离线环境保持可用

        x = self._signal
        n = int(x.shape[0])
        algo = str(self._params.get("algo", "pelt"))
        ms = max(2, int(self._params.get("min_size", max(2, round(0.01 * n)))))
        n_bkps = self._params.get("n_bkps")
        model = str(self._params.get("model", "l2"))
        if algo == "kernel":
            if n > _KERNEL_N_MAX:
                return []  # Gate C：超限拒绝执行，防 O(n^2) 爆内存
            kernel = str(self._params.get("kernel", "rbf"))
            seeker = rpt.KernelCPD(kernel=kernel, min_size=ms).fit(x)
            raw = (
                seeker.predict(pen=self._pen(x))
                if n_bkps is None
                else seeker.predict(n_bkps=int(n_bkps))
            )
        elif algo == "dynp":
            k = int(n_bkps) if n_bkps is not None else 1
            raw = rpt.Dynp(model=model, min_size=ms, jump=1).fit(x).predict(n_bkps=k)
        elif algo == "binseg":
            bs = rpt.Binseg(model=model, min_size=ms, jump=1).fit(x)
            raw = (
                bs.predict(pen=self._pen(x))
                if n_bkps is None
                else bs.predict(n_bkps=int(n_bkps))
            )
        elif algo == "window":
            wn = rpt.Window(model=model, width=max(4, 2 * ms), min_size=ms).fit(x)
            raw = (
                wn.predict(pen=self._pen(x))
                if n_bkps is None
                else wn.predict(n_bkps=int(n_bkps))
            )
        else:  # pelt
            pelt = rpt.Pelt(model=model, min_size=ms, jump=1).fit(x)
            raw = pelt.predict(pen=self._pen(x))
        # ruptures 输出为 [内部变化点..., n]（不含首 0）；实测末尾必含 n，
        # 去掉末尾并 int() 显式转换（Window 出 int64 / KernelCPD 出 int32）
        return [int(c) for c in raw[:-1]] if len(raw) >= 1 else []

    # -- 内部 -----------------------------------------------------------------
    def _pen(self, x: np.ndarray) -> float:
        """归一化惩罚 beta = lam * sigma_hat^2 * log n（与 numpy 兜底同口径）。"""
        from ..core.algorithms import beta_from_lambda

        lam = float(self._params.get("lam", 3.0))
        return beta_from_lambda(x, lam)


# --------------------------------------------------------------------------- 注册
_MODELS = ("l1", "l2", "normal", "rbf", "ar", "rank")


def _register_all() -> None:
    if not ruptures_available():
        return

    @register("ruptures-pelt-l2", doc="ruptures Pelt + l2（SOTA 主力）")
    def _p(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="pelt", model="l2", **p)

    @register("ruptures-pelt-l1", doc="ruptures Pelt + l1（重尾/离群鲁棒）")
    def _p1(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="pelt", model="l1", **p)

    @register("ruptures-pelt-normal", doc="ruptures Pelt + 高斯 NLL（均值+方差）")
    def _pn(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="pelt", model="normal", **p)

    @register("ruptures-pelt-rbf", doc="ruptures Pelt + rbf 核代价（任意分布变化）")
    def _pr(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="pelt", model="rbf", **p)

    @register("ruptures-pelt-ar", doc="ruptures Pelt + AR(p) 代价（谱/自相关变化）")
    def _pa(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="pelt", model="ar", **p)

    @register("ruptures-pelt-rank", doc="ruptures Pelt + rank 代价（非参数）")
    def _pk(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="pelt", model="rank", **p)

    @register("ruptures-binseg-l2", doc="ruptures Binseg + l2")
    def _b(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="binseg", model="l2", **p)

    @register("ruptures-kernel-rbf", doc="ruptures KernelCPD rbf（n<=5000，超限自动拒绝）")
    def _k(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="kernel", kernel="rbf", **p)

    @register("ruptures-dynp-l2", doc="ruptures Dynp 精确 DP（oracle-K 专用）")
    def _d(**p: Any) -> RupturesDetector:
        return RupturesDetector(algo="dynp", model="l2", **p)


_register_all()
