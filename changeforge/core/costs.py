"""l2 代价的前缀和实现与统计辅助。

数学核心（前缀和 O(1) 代价）::

    S1[k] = sum_{i<k} x_i          S2[k] = sum_{i<k} x_i^2
    c_l2(s, e) = (S2[e]-S2[s]) - (S1[e]-S1[s])^2 / (e-s)

ANOVA 分解恒等式（不变量 I2b，抓 off-by-one 的硬金标准）::

    c(s, e) = c(s, m) + c(m, e) + n1*n2/(n1+n2) * (mu1 - mu2)^2

为避免 |mean| >> sigma 时的灾难性相消，信号先减全局中位数
（平移不改变 l2 变点位置）。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

__all__ = ["PrefixCostL2", "mad_sigma", "anova_gap"]


class PrefixCostL2:
    """多元 l2 代价（各通道求和），前缀和 O(1) 查询。

    Args:
        signal: (n,) 或 (n, d) 数组；内部转为 (n, d) 并减全局中位数。
    """

    def __init__(self, signal: np.ndarray) -> None:
        x = np.asarray(signal, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] < 1:
            raise ValueError(f"signal 形状非法: {x.shape}")
        self.n = int(x.shape[0])
        # 平移不变：减中位数防灾难性相消（|x_mean| >> sigma 时）
        x = x - np.median(x, axis=0, keepdims=True)
        self._s1 = np.zeros((self.n + 1, x.shape[1]), dtype=np.float64)
        self._s2 = np.zeros((self.n + 1, x.shape[1]), dtype=np.float64)
        np.cumsum(x, axis=0, out=self._s1[1:])
        np.cumsum(x * x, axis=0, out=self._s2[1:])

    def cost(self, start: int, end: int) -> float:
        """段 [start, end) 的 l2 代价（多元为各通道之和）。"""
        if end - start < 1:
            raise ValueError(f"段长非法: [{start}, {end})")
        if end - start == 1:
            return 0.0  # 单点段代价为 0（I1）
        seg1 = self._s1[end] - self._s1[start]
        seg2 = self._s2[end] - self._s2[start]
        return float(np.sum(seg2 - (seg1 * seg1) / (end - start)))

    def mean(self, start: int, end: int) -> np.ndarray:
        return (self._s1[end] - self._s1[start]) / (end - start)


def mad_sigma(signal: np.ndarray) -> float:
    """MAD 鲁棒尺度估计 sigma_hat = 1.4826 * median(|x - median(x)|)。

    全链路共享同一 sigma_hat 估计器（跨算法公平对比的前提）。
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim > 1:
        x = x.reshape(-1)
    if x.size == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * float(mad)
    return sigma if sigma > 1e-12 else 1.0


def anova_gap(cost: PrefixCostL2, start: int, mid: int, end: int) -> float:
    """ANOVA 分解第三项 n1*n2/(n1+n2)*(mu1-mu2)^2（多通道取和）。

    满足恒等式 c(s,e) - c(s,m) - c(m,e) == anova_gap(s,m,e)。
    """
    n1 = mid - start
    n2 = end - mid
    mu1 = cost.mean(start, mid)
    mu2 = cost.mean(mid, end)
    return float(np.sum((n1 * n2) / (n1 + n2) * (mu1 - mu2) ** 2))
