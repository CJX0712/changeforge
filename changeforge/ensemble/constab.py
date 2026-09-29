"""ConStab-CPD 旗舰（CS-Gate）：共识—稳定门控变化点检测。

四阶段::

    S1 多视图候选生成   特征视图(原始 l2 / RFF-l2) x 惩罚轴(lam 网格) 的 PELT 候选
    S2 容差感知证据累积  S(t) = mean_views sum_tau tri_kernel_h(t - tau)，h = tol
    S3 跨惩罚稳定性选择  Pi(t) = 跨 lam 的视图出现频率；score = S * Pi^gamma
    S4 显著性闸门        循环块置换零分布 -> (1-alpha) 分位阈值 + NMS + conformal p

为什么能赢：真变点在宽 lam 区间稳定出现（Pi 高），假阳点只在个别 lam 偶然
出现（Pi 低）；软化证据带宽对齐评测容差（定位误差下降）；置换闸门把
null 假阳率压到 alpha 以内。

设计决策（实测依据，2026-09-29）:
    - 视图用自研向量化 PELT 而非 ruptures：本机 ruptures 无 Cython 扩展，
      单次 Pelt(l2, n=600) 约 1.1s，置换闸门需 B 次 x 视图数次调用，
      ruptures 后端在 20 视图 x 19 置换下单序列需 200s+，不可用；
    - 视图特征轴取「原始 l2 + RFF-l2」：RFF 通道标准化后 beta 尺度与
      原始 l2 同构（beta = lam*d*sigma^2*log n），跨视图公平；ruptures 的
      rbf/ar 代价尺度与 sigma^2*log n 不同构，混入会稀释证据且置换阈值失准
      （实测 n_perm=4 时全灭）；
    - 阈值取 np.quantile(null_max, 1-alpha, method="higher")：alpha=0.05
      需 n_perm >= 19 才有分辨率，默认 n_perm=19。

守护（Gate A/B/C）在基准层实现（examples/run_demo.py）：
    Gate A 校准集 FPR > alpha -> 收紧；Gate B 非劣回退 best_single；
    Gate C 预算闸门（RFF 视图 O(nD)，D=256 默认）。

作者: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.algorithms import (
    autocorr_block_len,
    beta_from_lambda,
    circular_block_permute,
    pelt_l2,
    rff_features,
)
from ..core.errors import InvalidDetectorParamsError
from ..core.types import Signal
from ..detectors import register

__all__ = ["ConStabCPD"]

_FEATURE_VIEWS = ("l2", "rff")


class ConStabCPD:
    """共识—稳定门控变化点检测器（ChangeForge 旗舰）。"""

    name = "constab-cpd"
    _allowed = (
        "lam_grid",
        "feature_views",
        "alpha",
        "gamma",
        "n_perm",
        "min_size",
        "tol",
        "seed",
        "rff_dim",
    )

    def __init__(self, **params: Any) -> None:
        self._params: dict[str, Any] = {}
        self._signal: np.ndarray | None = None
        self._gate_meta: dict[str, Any] = {}
        if params:
            self.set_params(**params)

    # -- 协议 ---------------------------------------------------------------
    def fit(self, signal: Signal) -> ConStabCPD:
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] < 6:
            raise InvalidDetectorParamsError("signal 至少需要 6 个样本", shape=arr.shape)
        self._signal = arr
        return self

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any) -> ConStabCPD:
        for key in params:
            if key not in self._allowed:
                raise InvalidDetectorParamsError(
                    "未知参数", detector=self.name, param=key, allowed=list(self._allowed)
                )
        self._params.update(params)
        return self

    # -- 主流程 ---------------------------------------------------------------
    def predict(self) -> list[int]:
        assert self._signal is not None
        x = self._signal
        n = int(x.shape[0])
        tol = max(1, int(self._params.get("tol", max(1, round(0.01 * n)))))
        ms = max(2, int(self._params.get("min_size", max(2, round(0.01 * n)))))
        alpha = float(self._params.get("alpha", 0.05))
        gamma = float(self._params.get("gamma", 1.0))
        n_perm = max(1, int(self._params.get("n_perm", 19)))
        seed = int(self._params.get("seed", 0))
        rff_dim = int(self._params.get("rff_dim", 256))
        lam_grid = tuple(self._params.get("lam_grid", (1.0, 2.0, 3.0, 5.0, 8.0)))
        views_spec = tuple(self._params.get("feature_views", _FEATURE_VIEWS))
        if not views_spec or not lam_grid:
            return []

        # 特征投影缓存（置换时 RFF 特征需随置换重算 —— 置换改变联合分布）
        def _project(sig: np.ndarray, view: str, rng: np.random.Generator) -> np.ndarray:
            if view == "l2":
                return sig
            if view == "rff":
                return rff_features(sig, n_features=rff_dim, seed=int(rng.integers(0, 2**31)))
            raise InvalidDetectorParamsError(
                "未知特征视图", detector=self.name, view=view, allowed=list(_FEATURE_VIEWS)
            )

        def _run_views(sig: np.ndarray, rng: np.random.Generator) -> dict:
            out: dict[tuple[str, float], list[int]] = {}
            for view in views_spec:
                feats = _project(sig, view, rng)
                for lam in lam_grid:
                    beta = beta_from_lambda(feats, lam)
                    out[(view, lam)] = pelt_l2(feats, beta, ms)
            return out

        def _score(sig: np.ndarray, views: dict, rng: np.random.Generator) -> np.ndarray:
            m = int(sig.shape[0])
            s = np.zeros(m, dtype=np.float64)
            for cps in views.values():
                for tau in cps:
                    lo = max(0, tau - tol)
                    hi = min(m, tau + tol + 1)
                    idx = np.arange(lo, hi)
                    s[idx] += 1.0 - np.abs(idx - tau) / tol
            s /= max(1, len(views))
            lam_means: list[np.ndarray] = []
            for lam in lam_grid:
                per_view: list[np.ndarray] = []
                for view in views_spec:
                    cps = views.get((view, lam), [])
                    ind = np.zeros(m, dtype=np.float64)
                    for tau in cps:
                        lo = max(0, tau - tol)
                        hi = min(m, tau + tol + 1)
                        ind[lo:hi] = 1.0
                    per_view.append(ind)
                if per_view:
                    lam_means.append(np.mean(per_view, axis=0))
            pi = np.mean(lam_means, axis=0) if lam_means else np.ones(m)
            return s * np.power(pi, gamma)

        rng = np.random.default_rng(seed)
        views = _run_views(x, rng)
        score = _score(x, views, rng)
        # S4 循环块置换零分布（块长 >= 自相关时间，保留时间结构）
        block = autocorr_block_len(x[:, 0])
        null_max: list[float] = []
        for _ in range(n_perm):
            perm = circular_block_permute(x, block, rng)
            null_max.append(float(np.max(_score(perm, _run_views(perm, rng), rng))))
        self._score_arr = score.copy()
        self._null_max_arr = list(null_max)
        threshold = float(np.quantile(null_max, 1.0 - alpha, method="higher"))

        # conformal 判据（正确的置换检验分辨率语义）：
        # p(t) = (1 + #{null_max >= score_t}) / (n_perm + 1) <= alpha 才报警。
        # 注意不能用 score >= quantile(1-alpha)：那允许 2 个 null 压过真实分，
        # 实际 FPR 系统性高一档（由 I12a 不变量实测抓出）。
        def _pval(sc: float) -> float:
            return (1 + sum(1 for nm in null_max if nm >= sc)) / (n_perm + 1)

        # 峰值提取：conformal p <= alpha 的候选按强度 NMS(min_size)
        cand = [t for t in range(ms, n - ms) if score[t] > 0 and _pval(score[t]) <= alpha]
        cand.sort(key=lambda t: (-score[t], t))
        picked: list[int] = []
        for t in cand:
            if all(abs(t - p) >= ms for p in picked):
                picked.append(int(t))
        picked.sort()
        # conformal p 值（审计与下游 FDR 用）
        pvals = [_pval(score[t]) for t in picked]
        self._gate_meta = {
            "threshold": threshold,
            "n_views": len(views),
            "n_perm": n_perm,
            "block_len": int(block),
            "alpha": alpha,
            "p_values": pvals,
            "scores": [float(score[t]) for t in picked],
            "backend": "numpy-vectorized-pelt",
        }
        return picked

    def last_gate_metrics(self) -> dict[str, Any]:
        """最近一次 predict 的闸门审计信息（写入 benchmark.json 用）。"""
        return dict(self._gate_meta) if self._gate_meta else {}


@register(
    "constab-cpd",
    doc="旗舰 ConStab-CPD：多视图证据累积 x 跨惩罚稳定性 x 置换显著性闸门",
)
def _build_constab(**params: Any) -> ConStabCPD:
    return ConStabCPD(**params)
