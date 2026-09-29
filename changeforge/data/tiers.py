"""五档难度梯度 DGP（用"可检测性指数"反解参数，禁止手调 Delta）。

可检测性指数::

    D = kappa - sqrt(2 * log(n / m_min)),  kappa = |Delta_mu| / (sigma * sqrt(1/m1+1/m2))

D < 0 统计上不可检测；0 < D < 1.5 难；1.5 < D < 3.5 中；D > 4 易。
DGP 只声明 D_target，由公式反解::

    Delta_mu / sigma = (D_target + sqrt(2*log(n/m_min))) * sqrt(1/m1 + 1/m2)

档位:
    D0  Smoke     K=1, n=1000, D=30（冒烟：min F1 >= 0.90 否则是实现 bug）
    D1  Null-iid  K=0, iid N(0,1)（门控方法的必胜档：假阳率 <= alpha）
    D1b Null-AR   K=0, AR(1) rho=0.7（诱饵档：自相关骗过 l2）
    D3  Medium    K=4, n=1000, D=1.7, 纯均值变化
    D4  Hard      K=8, 混合 4 类变化 + 2% 离群 + 2 诱饵段
    D5  Expert    K=20, n=2000, t(3) 重尾 + AR(0.6) + 3% 离群 + 4 诱饵段

seed = stable_hash(tier, idx)（sha256 前缀），全部走 np.random.default_rng，
禁用 np.random.seed（混用会破坏确定性不变量 I11）。

作者: 晨星
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.types import Dataset

__all__ = [
    "TIER_SPECS",
    "stable_seed",
    "detectability_delta",
    "make_tier_series",
    "TIER_ORDER",
]


def stable_seed(*parts: Any) -> int:
    """sha256 稳定哈希 -> 64 位正整数 seed（跨进程/跨平台一致）。"""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


@dataclass(frozen=True)
class TierSpec:
    tier: str
    n: int
    k: int
    d_target: float | None
    noise: str  # "normal" | "t3" | "gamma"
    ar_rho: float
    outlier_frac: float
    n_decoys: int
    change_types: tuple[str, ...]  # 变化类型轮转表
    is_null: bool = False
    note: str = ""


TIER_SPECS: dict[str, TierSpec] = {
    "D0": TierSpec(
        tier="D0",
        n=1000,
        k=1,
        d_target=30.0,
        noise="normal",
        ar_rho=0.0,
        outlier_frac=0.0,
        n_decoys=0,
        change_types=("mean",),
        note="冒烟档：min(F1) >= 0.90，否则是实现 bug 而非难度问题",
    ),
    "D1": TierSpec(
        tier="D1",
        n=1000,
        k=0,
        d_target=None,
        noise="normal",
        ar_rho=0.0,
        outlier_frac=0.0,
        n_decoys=0,
        change_types=(),
        is_null=True,
        note="Null-iid：门控方法假阳率必须 <= alpha",
    ),
    "D1b": TierSpec(
        tier="D1b",
        n=1000,
        k=0,
        d_target=None,
        noise="normal",
        ar_rho=0.7,
        outlier_frac=0.0,
        n_decoys=0,
        change_types=(),
        is_null=True,
        note="Null-AR(0.7)：自相关诱骗 l2 型代价，拉开方法差距",
    ),
    "D3": TierSpec(
        tier="D3",
        n=1000,
        k=4,
        d_target=1.7,
        noise="normal",
        ar_rho=0.0,
        outlier_frac=0.0,
        n_decoys=0,
        change_types=("mean",),
        note="主战场：纯均值变化（l2 类代价唯一能稳定看见的类型）；"
        "混合类型档 D4/D5 的 F1 天然被方差/AR 边界稀释，跨档不可直接比",
    ),
    "D4": TierSpec(
        tier="D4",
        n=1000,
        k=8,
        d_target=1.0,
        noise="normal",
        ar_rho=0.0,
        outlier_frac=0.02,
        n_decoys=2,
        change_types=("mean", "var", "ar", "dist"),
        note="混合变化档：视图互补性主证明（2均值+2方差+2AR+2族切换）",
    ),
    "D5": TierSpec(
        tier="D5",
        n=2000,
        k=20,
        d_target=0.5,
        noise="t3",
        ar_rho=0.6,
        outlier_frac=0.03,
        n_decoys=4,
        change_types=("mean", "var", "ar", "dist"),
        note="极端档：重尾+自相关+离群+诱饵，允许大家一般但方法间须拉开 >= 0.15",
    ),
}

TIER_ORDER = ("D0", "D1", "D1b", "D3", "D4", "D5")


def detectability_delta(n: int, m_min: int, d_target: float) -> float:
    """由可检测性指数反解 Delta_mu / sigma（公式见模块 docstring）。

    注意：本函数按**单一 m_min** 标定，仅适用于段长均匀的档位（如 D3）。
    段长不均匀的档位（D4/D5）必须改用 :func:`detectability_delta_pair`，
    按相邻两段的实际长度对逐边界反解——否则长段会被过度加强，
    造成"名义 D_target 很小、实际极易检测"的难度倒挂。
    """
    extreme = float(np.sqrt(2.0 * np.log(n / m_min)))
    return float((d_target + extreme) * np.sqrt(1.0 / m_min + 1.0 / m_min))


def detectability_delta_pair(n: int, m1: int, m2: int, d_target: float) -> float:
    """按相邻两段的实际长度对反解 Delta_mu / sigma。

        kappa = |Delta_mu| / (sigma * sqrt(1/m1 + 1/m2))
        D     = kappa - sqrt(2 * log(n / min(m1, m2)))
        => Delta_mu/sigma = (D_target + sqrt(2*log(n/min(m1,m2)))) * sqrt(1/m1 + 1/m2)

    段长不均匀时（D4/D5 随机段长）必须用这个版本，每个边界才都能精确
    落在目标可检测性上，难度才随 D_target 单调。
    """
    m_lo = max(1, min(m1, m2))
    extreme = float(np.sqrt(2.0 * np.log(max(2.0, n / m_lo))))
    return float((d_target + extreme) * np.sqrt(1.0 / m1 + 1.0 / m2))


def _noise(kind: str, rng: np.random.Generator, size: int) -> np.ndarray:
    if kind == "t3":
        return rng.standard_t(3.0, size=size)
    if kind == "gamma":
        return rng.gamma(2.0, 1.0, size=size) - 2.0
    return rng.standard_normal(size=size)


def _ar_noise(rng: np.random.Generator, size: int, rho: float, base: np.ndarray) -> np.ndarray:
    """AR(1) 滤波：x_t = rho*x_{t-1} + e_t，e 取 base（保持噪声族语义）。"""
    if rho <= 0:
        return base
    out = np.empty_like(base)
    prev = 0.0
    for i in range(base.shape[0]):
        prev = rho * prev + base[i]
        out[i] = prev
    return out


def make_tier_series(tier: str, idx: int) -> Dataset:
    """按档位规格生成第 idx 条确定性合成序列。"""
    spec = TIER_SPECS[tier]
    n, k = spec.n, spec.k
    rng = np.random.default_rng(stable_seed("changeforge", tier, idx))
    name = f"{tier}-{idx:03d}"

    # ---- Null 档 -----------------------------------------------------------
    if spec.is_null:
        base = _noise(spec.noise, rng, n)
        sig = _ar_noise(rng, n, spec.ar_rho, base)
        return Dataset(
            name=name,
            signal=sig,
            change_points=[],
            metadata={"tier": tier, "n": n, "k": 0, "note": spec.note},
        )

    # ---- 分段边界 ------------------------------------------------------------
    if tier == "D0":
        bounds = [0, n // 2, n]
    elif tier == "D3":
        bounds = [0, 200, 400, 600, 800, n]
    else:
        m_min = {"D4": 60, "D5": 20}[tier]
        # 随机段长 >= m_min，总和 = n（拒绝采样保证终止）
        lengths: list[int] = []
        remaining = n
        for i in range(k, 0, -1):
            floor = m_min
            ceil = max(m_min, remaining - i * m_min)
            lengths.append(int(rng.integers(floor, ceil + 1)))
            remaining -= lengths[-1]
        lengths.append(remaining)
        bounds: list[int] = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        bounds[-1] = n
    segments = [(bounds[j], bounds[j + 1]) for j in range(len(bounds) - 1)]
    interior = bounds[1:-1]

    # ---- 每段参数 + 变化类型轮转 ----------------------------------------------
    m_min_eff = min(e - s for s, e in segments)
    sig = np.zeros(n, dtype=np.float64)
    state_mu = 0.0
    state_sd = 1.0
    state_rho = 0.0
    state_dist = "normal"
    for j, (s, e) in enumerate(segments):
        if j > 0:
            ctype = spec.change_types[(j - 1) % len(spec.change_types)]
            if ctype == "mean":
                m_prev = segments[j - 1][1] - segments[j - 1][0]
                m_cur = e - s
                state_mu += detectability_delta_pair(n, m_prev, m_cur, spec.d_target) * state_sd
            elif ctype == "var":
                # 交替而非累积：连乘会让 sd 随 K 指数放大（K=20 时达 10.5σ），
                # 造成"变点越多反而越好检"的难度倒挂。
                state_sd = 1.35 if state_sd < 1.2 else 1.0
            elif ctype == "ar":
                state_rho = 0.40 if state_rho < 0.25 else 0.15
            elif ctype == "dist":
                state_dist = "gamma" if state_dist == "normal" else "normal"
        m = e - s
        base = _noise(state_dist if state_dist != "gamma" else "gamma", rng, m)
        seg = state_mu + state_sd * _ar_noise(rng, m, spec.ar_rho + state_rho, base)
        sig[s:e] = seg

    # ---- 离群污染 --------------------------------------------------------------
    if spec.outlier_frac > 0:
        n_out = int(spec.outlier_frac * n)
        if n_out > 0:
            pos = rng.choice(n, size=n_out, replace=False)
            amp = rng.choice([-1.0, 1.0], size=n_out) * rng.uniform(5.0, 8.0, size=n_out)
            sig[pos] += amp

    # ---- 诱饵段边界（分布相同的假边界，专治假阳，不计入真值） --------------------
    decoys: list[int] = []
    for j in range(spec.n_decoys):
        seg_idx = int(rng.integers(0, len(segments)))
        s, e = segments[seg_idx]
        if e - s >= 4:
            decoys.append(int((s + e) // 2))
    decoys = sorted(d for d in decoys if all(abs(d - c) > m_min_eff for c in interior))

    return Dataset(
        name=name,
        signal=sig,
        change_points=[int(c) for c in interior],
        metadata={
            "tier": tier,
            "n": n,
            "k": k,
            "d_target": spec.d_target,
            "decoys": decoys,
            "note": spec.note,
        },
    )
