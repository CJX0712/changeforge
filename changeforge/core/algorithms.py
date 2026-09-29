"""纯 numpy 变化点算法内核：PELT / Dynp / BinSeg / Window / CUSUM / 滑动 Welch-t / RFF。

约定（全项目统一）::

    变化点 tau 表示新段起始索引，tau_0 = 0，tau_{K+1} = n，段 = [tau_k, tau_{k+1})
    返回值为内部变化点列表（不含 0 与 n），严格递增、int 类型。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from .costs import PrefixCostL2, mad_sigma

__all__ = [
    "pelt_l2",
    "dynp_l2",
    "binseg_l2",
    "window_l2",
    "cusum_online",
    "sliding_welch_t",
    "rff_features",
    "circular_block_permute",
    "autocorr_block_len",
    "beta_from_lambda",
]


def beta_from_lambda(signal: np.ndarray, lam: float) -> float:
    """归一化惩罚 beta = lam * d * sigma_hat^2 * log n。

    多通道 l2 代价是各通道之和（尺度 ~ d），beta 必须同尺度（乘 d）。
    I10 仿射不变性的根据：sigma_hat 随 a*x+b 中 a 线性缩放，
    beta 随 a^2 缩放，与 l2 代价同尺度，故变点集不变。
    """
    x = np.asarray(signal, dtype=np.float64)
    n = max(int(x.shape[0]), 2)
    sigma = mad_sigma(x)
    d = 1 if x.ndim == 1 else int(x.shape[1])
    return float(lam * d * sigma * sigma * np.log(n))


def _backtrack(parent: np.ndarray, n: int) -> list[int]:
    cps: list[int] = []
    t = n
    while parent[t] > 0:
        cps.append(int(parent[t]))
        t = int(parent[t])
    return sorted(set(cps))


def pelt_l2(
    signal: np.ndarray, beta: float, min_size: int = 10, prune: bool = True
) -> list[int]:
    """PELT（Killick et al. 2012），l2 代价 + 前缀和，候选步内向量化。

    递推::

        F[t] = min_{tau <= t-min_size} F[tau] + c(tau, t) + beta,  F[0] = -beta

    剪枝（Thm 3.1 正确形式）：若存在 s ∈ (tau, t) 使
    F(tau) + c(tau, s) + beta >= F(s)，
    即「经 tau 到达 s 不优于到达 s 的最优方式」，则 tau 对所有 T >= t
    都不可能最优，永久剔除（证明：c(tau,T) >= c(tau,s)+c(s,T) 超可加性 +
    上式链式放大）。实现取 s = best_tau（当届最优前驱），仅对 tau < s 检查，
    属保守剪枝（可能漏剪，绝不误剪，全局最优性保持）。
    注意：错误条件 c(tau,t) >= c(tau,s)+c(s,t)+beta 在 beta=0 时对 l2 恒成立
    （ANOVA 超可加），会把候选剪光退化成贪心链——由 I4 不变量实测抓出。
    beta=0 时正确条件下剪枝最强 => 结果等价无惩罚最优分割（I4）。
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    min_size = max(2, int(min_size))
    if n < 2 * min_size:
        return []
    x = x - np.median(x, axis=0, keepdims=True)  # 平移不变，防灾难性相消
    s1 = np.zeros((n + 1, d), dtype=np.float64)
    s2 = np.zeros((n + 1, d), dtype=np.float64)
    np.cumsum(x, axis=0, out=s1[1:])
    np.cumsum(x * x, axis=0, out=s2[1:])
    big = np.inf
    f = np.full(n + 1, big)
    parent = np.zeros(n + 1, dtype=np.int64)
    f[0] = -beta
    cand = np.zeros(1, dtype=np.int64)  # 有序候选集
    for t in range(min_size, n + 1):
        eligible = cand[cand <= t - min_size]
        if eligible.size == 0:
            f[t] = big
            parent[t] = 0
            continue
        seg1 = s1[t] - s1[eligible]  # (k, d)
        seg2 = s2[t] - s2[eligible]
        lens = (t - eligible).astype(np.float64)  # (k,)
        costs = np.sum(seg2 - (seg1 * seg1) / lens[:, None], axis=1)
        vals = f[eligible] + costs + beta
        j = int(np.argmin(vals))
        best_tau = int(eligible[j])
        f[t] = float(vals[j])
        parent[t] = best_tau
        if prune and best_tau > 0:
            keep = np.ones(cand.size, dtype=bool)
            elig_mask = cand <= t - min_size
            lt_mask = elig_mask & (cand < best_tau)
            if np.any(lt_mask):
                taus = cand[lt_mask]
                sg1 = s1[best_tau] - s1[taus]
                sg2 = s2[best_tau] - s2[taus]
                lb = (best_tau - taus).astype(np.float64)
                c_tb = np.sum(sg2 - (sg1 * sg1) / lb[:, None], axis=1)
                # 正确剪枝条件: F(tau) + c(tau, s) + beta >= F(s), s = best_tau
                prunable_local = f[taus] + c_tb + beta >= f[best_tau]
                keep[np.where(lt_mask)[0]] &= ~prunable_local
            cand = cand[keep]
        cand = np.append(cand, np.int64(t))
    return _backtrack(parent, n)


def dynp_l2(signal: np.ndarray, n_bkps: int, min_size: int = 2) -> list[int]:
    """精确动态规划（给定 K 的全局最优），O(K n^2)，仅用于小 n 与 oracle 对照。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = int(x.shape[0])
    k = int(n_bkps)
    min_size = max(1, int(min_size))
    if k <= 0 or n < (k + 1) * min_size:
        return []
    cost = PrefixCostL2(x)
    big = np.inf
    g = np.full((k + 1, n + 1), big)
    par = np.zeros((k + 1, n + 1), dtype=np.int64)
    for t in range(min_size, n + 1):
        g[0][t] = cost.cost(0, t)
    for j in range(1, k + 1):
        for t in range((j + 1) * min_size, n + 1):
            best = big
            best_tau = -1
            for tau in range(j * min_size, t - min_size + 1):
                val = g[j - 1][tau] + cost.cost(tau, t)
                if val < best:
                    best = val
                    best_tau = tau
            g[j][t] = best
            par[j][t] = best_tau
    cps: list[int] = []
    t = n
    for j in range(k, 0, -1):
        tau = int(par[j][t])
        cps.append(tau)
        t = tau
    return sorted(int(c) for c in cps if 0 < c < n)


def binseg_l2(
    signal: np.ndarray, n_bkps: int | None = None, pen: float = 0.0, min_size: int = 10
) -> tuple[list[int], list[float]]:
    """二分分割（Scott & Knott 1974）。返回 (内部变化点, 各轮增益)。

    增益 = c(a,b) - c(a,m) - c(m,b) >= 0（ANOVA 恒等式）；
    gains 序列非严格递减、总代价随 K 单调不增（I6）。
    n_bkps 模式切满 K 刀；pen 模式在 gain <= pen 时停止。
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = int(x.shape[0])
    min_size = max(2, int(min_size))
    cost = PrefixCostL2(x)
    intervals: list[tuple[int, int]] = [(0, n)]
    cps: list[int] = []
    gains: list[float] = []
    while intervals:
        best_gain = -np.inf
        best_t = -1
        best_idx = -1
        for idx, (a, b) in enumerate(intervals):
            if b - a < 2 * min_size:
                continue
            base = cost.cost(a, b)
            for m in range(a + min_size, b - min_size + 1):
                gain = base - cost.cost(a, m) - cost.cost(m, b)
                if gain > best_gain:
                    best_gain = float(gain)
                    best_t = m
                    best_idx = idx
        if best_t < 0:
            break
        if n_bkps is None and best_gain <= pen:
            break
        if n_bkps is not None and len(cps) >= n_bkps:
            break
        cps.append(int(best_t))
        gains.append(best_gain)
        a, b = intervals.pop(best_idx)
        intervals.append((a, best_t))
        intervals.append((best_t, b))
    return sorted(cps), gains


def window_l2(
    signal: np.ndarray,
    width: int = 50,
    min_size: int = 10,
    n_bkps: int | None = None,
    pen: float | None = None,
) -> list[int]:
    """滑动窗口：score(t) = c(t-W,t)+c(t,t+W)-c(t-W,t+W)，局部峰 + NMS。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = int(x.shape[0])
    w = max(5, int(width))
    cost = PrefixCostL2(x)
    scores = np.zeros(n, dtype=np.float64)
    lo = w
    hi = n - w
    if hi > lo:
        idx = np.arange(lo, hi)
        left = np.array([cost.cost(t - w, t) for t in idx])
        right = np.array([cost.cost(t, t + w) for t in idx])
        whole = np.array([cost.cost(t - w, t + w) for t in idx])
        scores[idx] = whole - left - right  # 分割增益，越大越好（ruptures Window 同语义）
    cand = [t for t in range(lo, hi) if scores[t] > 0]
    if pen is not None:
        cand = [t for t in cand if scores[t] > pen]
    cand.sort(key=lambda t: (-scores[t], t))
    picked: list[int] = []
    for t in cand:
        if all(abs(t - p) >= max(1, min_size) for p in picked):
            picked.append(t)
        if n_bkps is not None and len(picked) >= n_bkps:
            break
    return sorted(picked)


def cusum_online(
    signal: np.ndarray,
    delta: float | None = None,
    h: float = 5.0,
    burn: int = 50,
    cooldown: int = 25,
) -> list[int]:
    """双侧 CUSUM（Page 1954），在线均值漂移检测。

    增量 s_t = (delta/sigma^2) * (x_t - mu0 - delta/2)；g = max(0, g + s_t)；
    g >= h 报警后复位。mu0/sigma 由前 burn 个样本估计（MAD）。
    报警后进入 cooldown 冷却期（持续漂移段防连环误报），且报警间隔 >= cooldown。
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 3 * max(2, burn // 2):
        return []
    nb = min(burn, max(10, n // 4))
    mu0 = float(np.mean(x[:nb]))
    sigma = mad_sigma(x[:nb])
    if delta is None:
        delta = 1.0 * sigma
    cd = max(1, int(cooldown))
    alarms: list[int] = []
    for direction in (+1.0, -1.0):
        d = direction * delta
        g = 0.0
        quiet = 0
        for t in range(nb, n):
            s = (d / (sigma * sigma)) * (x[t] - mu0 - d / 2.0)
            g = max(0.0, g + s)
            quiet += 1
            if g >= h and quiet >= cd:
                alarms.append(int(t))
                g = 0.0
                quiet = 0
                # 重新基线：报警后以最近 cd 个样本的均值作为新基准，
                # 否则持续漂移段每过 cooldown 必连环报警
                mu0 = float(np.mean(x[max(nb, t - cd) : t + 1]))
    # 两个方向的报警按时间合并（间隔 < cooldown 只留最先）
    alarms.sort()
    merged: list[int] = []
    for a in alarms:
        if all(abs(a - m) >= cd for m in merged):
            merged.append(a)
    return merged


def sliding_welch_t(
    signal: np.ndarray, window: int = 50, alpha: float = 0.01, min_size: int = 10
) -> list[int]:
    """滑动 Welch-t + BH 多重检验校正 + 峰值 NMS。

    约 n 次检验，不做 BH 校正假阳必爆表（历史高频翻车点）。
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = int(x.size)
    w = max(5, int(window))
    if n < 2 * w + 2:
        return []
    ts = np.zeros(n, dtype=np.float64)
    lo, hi = w, n - w
    for t in range(lo, hi):
        a = x[t - w : t]
        b = x[t : t + w]
        va = float(np.var(a, ddof=1))
        vb = float(np.var(b, ddof=1))
        if va + vb <= 1e-15:
            ts[t] = 0.0
            continue
        se = np.sqrt(va / w + vb / w)
        ts[t] = abs(float(np.mean(a) - np.mean(b))) / float(se)
    # 正态近似 p 值 + BH 校正
    from math import erf, sqrt

    def sf(z: float) -> float:  # 标准正态尾概率
        return 0.5 * (1.0 - erf(z / sqrt(2.0)))

    positions = [t for t in range(lo, hi) if ts[t] > 0]
    if not positions:
        return []
    pvals = [min(1.0, 2.0 * sf(float(ts[t]))) for t in positions]
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 1.0
    for rank, i in enumerate(reversed(order)):
        k = m - rank
        running = min(running, pvals[i] * m / k)
        adj[i] = running
    cand = [positions[i] for i in range(m) if adj[i] < alpha]
    cand.sort(key=lambda t: (-ts[t], t))
    picked: list[int] = []
    nms_radius = max(1, min_size, w // 2)  # 跨窗重复显著性需大抑制半径
    for t in cand:
        if all(abs(t - p) >= nms_radius for p in picked):
            picked.append(t)
    return sorted(picked)


def rff_features(signal: np.ndarray, n_features: int = 256, seed: int = 0) -> np.ndarray:
    """随机傅里叶特征逼近 RBF 核（KernelCPD 的 O(nD) 替代，也是兜底核视图）。

    带宽用 median heuristic，且只在 <= 2000 点的子样本上算（防 O(n^2)）。
    输出通道标准化到单位方差，使 beta 口径与单通道一致。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    sub = x[rng.choice(n, size=min(n, 2000), replace=False)] if n > 2000 else x
    # median heuristic: subsample 两两距离的中位数
    if sub.shape[0] > 512:
        ii = rng.choice(sub.shape[0], size=512, replace=False)
        sub = sub[ii]
    diffs = sub[:, None, :] - sub[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    med = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    sigma = med if med > 1e-12 else 1.0
    d_feat = max(16, int(n_features))
    w = rng.normal(0.0, 1.0 / sigma, size=(d, d_feat))
    b = rng.uniform(0.0, 2.0 * np.pi, size=d_feat)
    z = np.sqrt(2.0 / d_feat) * np.cos(x @ w + b)
    std = z.std(axis=0)
    std[std < 1e-12] = 1.0
    return z / std


def autocorr_block_len(signal: np.ndarray, max_lag: int = 50) -> int:
    """估计自相关时间（ACF 首次跌破 0.2 的滞后），截断到 [1, max_lag]。"""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = int(x.size)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12 or n < 4:
        return 1
    for lag in range(1, min(max_lag, n // 4) + 1):
        num = float(np.dot(x[:-lag], x[lag:]))
        if num / denom < 0.2:
            return max(1, lag)
    return max(1, min(max_lag, n // 4))


def circular_block_permute(
    signal: np.ndarray, block_len: int, rng: np.random.Generator
) -> np.ndarray:
    """循环块置换（标准定义）：按块切块 -> 块内保序 -> 块间随机洗牌。

    保留局部自相关结构（长度 >= block_len 的片段原样保留），且严格保持
    multiset（排序后逐位相等，可用作零分布生成器）。
    注意：随机起点取块（block bootstrap）不是置换，会重复/遗漏元素。
    """
    x = np.asarray(signal)
    n = int(x.shape[0])
    b = max(1, int(block_len))
    n_blocks = int(np.ceil(n / b))
    order = rng.permutation(n_blocks)
    idx = np.concatenate([np.arange(int(i) * b, min((int(i) + 1) * b, n)) for i in order])
    return x[idx[:n]]
