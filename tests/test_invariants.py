"""核心不变量单测（I1–I16 精选实现，每条注明抓的是什么 bug）。

I1  代价非负与零性          -> 代价定义错 / 归一化错
I2a 前缀和 ≡ 直接计算       -> off-by-one / 求和索引错
I2b ANOVA 分解恒等式        -> 段边界语义错（最常踩）
I3  Dynp ≡ 穷举最优         -> DP 递推/初始化错
I4  PELT(β=0) ≡ Dynp(K*)   -> 剪枝把最优解剪掉了（PELT 最致命 bug）
I5  PELT(β>0) ≡ 带罚暴力    -> 惩罚项位置错
I6  BinSeg gains 递减       -> 贪心选择错 / 区间维护错
I7  输出格式合法性          -> JSON 序列化炸 / 下游匹配炸
I8  β 递增 => K 非增        -> 剪枝与惩罚不一致
I9a numpy-PELT ≡ ruptures   -> 兜底与后端语义偏离
I10 β 归一化仿射不变性      -> 跨算法对比失去意义
I11 同 seed 逐位一致        -> 随机源/集合迭代顺序引入非确定
I12a Null 档 FPR <= alpha   -> 置换零分布失效

作者: 晨星
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from changeforge.core.algorithms import (
    beta_from_lambda,
    binseg_l2,
    circular_block_permute,
    dynp_l2,
    pelt_l2,
)
from changeforge.core.costs import PrefixCostL2, anova_gap
from changeforge.data.tiers import TIER_ORDER, make_tier_series, stable_seed
from changeforge.detectors import predict_with


def _seg_signal(rng: np.random.Generator, n_seg: int = 5, seg: int = 60) -> np.ndarray:
    mus = rng.uniform(-3, 3, size=n_seg)
    return np.concatenate([rng.normal(mu, 1.0, size=seg) for mu in mus]).reshape(-1, 1)


# --------------------------------------------------------------------------- I1
def test_i1_cost_nonnegative_and_zero_on_constant():
    rng = np.random.default_rng(1)
    x = _seg_signal(rng, 4, 50)
    cost = PrefixCostL2(x)
    for _ in range(50):
        s = int(rng.integers(0, x.shape[0] - 2))
        e = int(rng.integers(s + 2, x.shape[0] + 1))
        assert cost.cost(s, e) >= -1e-12
    const = np.full((80, 1), 3.14)
    cc = PrefixCostL2(const)
    assert abs(cc.cost(0, 80)) < 1e-9
    assert cc.cost(5, 6) == 0.0  # 单点段


# --------------------------------------------------------------------------- I2a
def test_i2a_prefix_matches_direct():
    rng = np.random.default_rng(2)
    x = _seg_signal(rng, 4, 40)
    cost = PrefixCostL2(x)
    x1 = x - np.median(x, axis=0, keepdims=True)
    for _ in range(200):
        s = int(rng.integers(0, x.shape[0] - 1))
        e = int(rng.integers(s + 1, x.shape[0] + 1))
        direct = float(np.sum((x1[s:e] - x1[s:e].mean(axis=0)) ** 2))
        assert abs(cost.cost(s, e) - direct) <= 1e-9 * max(1.0, direct)


# --------------------------------------------------------------------------- I2b
def test_i2b_anova_identity():
    """c(s,e) == c(s,m) + c(m,e) + n1*n2/(n1+n2)*(mu1-mu2)^2 —— 抓 off-by-one。"""
    rng = np.random.default_rng(3)
    x = _seg_signal(rng, 6, 30)
    cost = PrefixCostL2(x)
    for _ in range(300):
        s = int(rng.integers(0, x.shape[0] - 2))
        m = int(rng.integers(s + 1, x.shape[0] - 1))
        e = int(rng.integers(m + 1, x.shape[0] + 1))
        lhs = cost.cost(s, e) - cost.cost(s, m) - cost.cost(m, e)
        rhs = anova_gap(cost, s, m, e)
        assert abs(lhs - rhs) <= 1e-9 * max(1.0, abs(rhs))


# --------------------------------------------------------------------------- I3
def test_i3_dynp_equals_bruteforce():
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        n, k, ms = 12, 3, 2
        x = rng.normal(0, 1, size=n)
        x[rng.integers(0, n)] += 5.0
        x[rng.integers(0, n)] += 5.0
        dp_cost, dp_cps = _dynp_total_cost(x, k, ms)
        best_brute = np.inf
        for combo in itertools.combinations(range(1, n), k):
            bounds = (0,) + combo + (n,)
            if any(b2 - b1 < ms for b1, b2 in zip(bounds, bounds[1:])):
                continue
            c = _total_cost_direct(x, bounds)
            best_brute = min(best_brute, c)
        assert best_brute < np.inf
        assert abs(dp_cost - best_brute) <= 1e-9 * max(1.0, best_brute)
        assert all(0 < c < n for c in dp_cps)


def _dynp_total_cost(x: np.ndarray, k: int, ms: int) -> tuple[float, list[int]]:
    cps = dynp_l2(x.reshape(-1, 1), k, ms)
    bounds = (0,) + tuple(cps) + (x.shape[0],)
    return _total_cost_direct(x, bounds), cps


def _total_cost_direct(x: np.ndarray, bounds: tuple[int, ...]) -> float:
    total = 0.0
    for a, b in zip(bounds, bounds[1:]):
        seg = x[a:b]
        total += float(np.sum((seg - seg.mean()) ** 2))
    return total


# --------------------------------------------------------------------------- I4
def test_i4_pelt_beta0_matches_dynp():
    """β=0 时 PELT 剪枝最强，总代价必须等于 Dynp(K*) 的全局最优。"""
    for seed in range(3):
        rng = np.random.default_rng(200 + seed)
        x = _seg_signal(rng, 5, 25)
        n = x.shape[0]
        ms = 5
        pelt_cps = pelt_l2(x, beta=0.0, min_size=ms)
        f_cost = _total_cost_direct(x[:, 0], (0,) + tuple(pelt_cps) + (n,))
        best = np.inf
        for k in range(0, n // ms):
            cps = dynp_l2(x, k, ms)
            c = _total_cost_direct(x[:, 0], (0,) + tuple(cps) + (n,))
            best = min(best, c)
        assert abs(f_cost - best) <= 1e-9 * max(1.0, best), (f_cost, best)


# --------------------------------------------------------------------------- I5
def test_i5_pelt_penalty_matches_bruteforce():
    """PELT(β>0) 的目标值 == 穷举所有子集的 min(代价 + βK)（n<=14）。"""
    for seed in range(2):
        rng = np.random.default_rng(300 + seed)
        n, ms = 14, 2
        x = np.concatenate([rng.normal(0, 1, 7), rng.normal(2, 1, 7)]).reshape(-1, 1)
        beta = 4.0
        pelt_cps = pelt_l2(x, beta=beta, min_size=ms)
        pelt_obj = _total_cost_direct(x[:, 0], (0,) + tuple(pelt_cps) + (n,)) + beta * len(
            pelt_cps
        )
        best = np.inf
        for k in range(0, n):
            for combo in itertools.combinations(range(1, n), k):
                bounds = (0,) + combo + (n,)
                if any(b2 - b1 < ms for b1, b2 in zip(bounds, bounds[1:])):
                    continue
                best = min(best, _total_cost_direct(x[:, 0], bounds) + beta * k)
        assert best < np.inf
        assert pelt_obj <= best + 1e-9, (pelt_obj, best)
        assert pelt_obj >= best - 1e-9, (pelt_obj, best)  # PELT 全局最优


# --------------------------------------------------------------------------- I6
def test_i6_binseg_cost_monotone():
    """BinSeg 正确不变量：gains >= 0（ANOVA），且累计增益 == 总代价下降量。

    注：「逐轮增益非递增」不是 BinSeg 的定理——区间细分后子区间内的增益
    按新基线重算，可大于上一轮次优（诚实修正，不硬凑原 I6 表述）。
    """
    from changeforge.core.costs import PrefixCostL2

    for seed in range(3):
        rng = np.random.default_rng(400 + seed)
        x = _seg_signal(rng, 6, 30)
        cps, gains = binseg_l2(x, n_bkps=6, min_size=5)
        assert len(gains) >= 2
        assert all(g >= -1e-12 for g in gains)  # ANOVA gap >= 0
        # 总代价下降 == 累计增益（每轮接受分裂使总代价减去该轮增益）
        cost = PrefixCostL2(x)
        total0 = cost.cost(0, x.shape[0])
        bounds = (0,) + tuple(cps) + (x.shape[0],)
        total_k = sum(cost.cost(a, b) for a, b in zip(bounds, bounds[1:]))
        assert abs((total0 - total_k) - sum(gains)) <= 1e-8 * max(1.0, total0)


def _small_seg_signal(rng: np.random.Generator, n_seg: int = 4, seg: int = 100) -> np.ndarray:
    """小规模带变点信号（格式/一致性类测试用，控制 ruptures 纯 Python 耗时）。"""
    mus = rng.uniform(-2, 2, size=n_seg)
    return np.concatenate([rng.normal(mu, 1.0, size=seg) for mu in mus]).reshape(-1, 1)


# --------------------------------------------------------------------------- I7
@pytest.mark.parametrize(
    "detector",
    [
        "numpy-pelt-l2",
        "numpy-binseg-l2",
        "numpy-window-l2",
        "numpy-mmd-rff",
        "ruptures-pelt-l2",
        "ruptures-pelt-rank",
        "ruptures-kernel-rbf",
    ],
)
def test_i7_output_format(detector):
    for idx in range(2):
        rng = np.random.default_rng(700 + idx)
        x = _small_seg_signal(rng, 4, 100)
        cps = predict_with(detector, x, min_size=10)
        n = int(x.shape[0])
        assert all(isinstance(c, int) for c in cps)  # 非 np.int64/int32
        assert all(1 <= c <= n - 1 for c in cps)
        assert all(b - a >= 1 for a, b in zip(cps, cps[1:]))  # 严格递增去重
        assert len(cps) <= n // 10  # K <= n/min_size - 1


# --------------------------------------------------------------------------- I8
def test_i8_beta_monotone_k():
    rng = np.random.default_rng(500)
    x = _seg_signal(rng, 6, 40)
    counts = []
    for lam in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        beta = beta_from_lambda(x, lam)
        counts.append(len(pelt_l2(x, beta, min_size=10)))
    assert all(b <= a for a, b in zip(counts, counts[1:]))  # 非增（允许平台）


# --------------------------------------------------------------------------- I9a
def test_i9a_numpy_pelt_matches_ruptures():
    import ruptures as rpt

    agree = 0
    total = 0
    for idx in range(8):
        rng = np.random.default_rng(900 + idx)
        x = _small_seg_signal(rng, 5, 80)
        ms = 10
        lam = 3.0
        beta = beta_from_lambda(x, lam)
        mine = pelt_l2(x, beta, ms)
        raw = rpt.Pelt(model="l2", min_size=ms, jump=1).fit(x).predict(pen=beta)
        theirs = [int(c) for c in raw[:-1]]
        total += 1
        # 定位允许 Hausdorff <= 2（内部实现细节差异），集合大小应一致
        if len(mine) == len(theirs):
            if not mine or max(abs(a - b) for a, b in zip(mine, theirs)) <= 2:
                agree += 1
    assert agree >= total * 0.8, (agree, total)


# --------------------------------------------------------------------------- I10
def test_i10_affine_invariance():
    for idx in range(4):
        ds = make_tier_series("D3", idx)
        base = predict_with("numpy-pelt-l2", ds.signal, min_size=10)
        for a, b in ((0.1, 0.0), (10.0, -5.0), (2.5, 3.0)):
            scaled = a * ds.signal + b
            got = predict_with("numpy-pelt-l2", scaled, min_size=10)
            assert got == base, (a, b, got, base)


# --------------------------------------------------------------------------- I11
def test_i11_bitwise_deterministic():
    ds = make_tier_series("D4", 0)
    import hashlib

    def _hash(cps):
        return hashlib.sha256(",".join(map(str, cps)).encode()).hexdigest()

    h1 = _hash(predict_with("numpy-pelt-l2", ds.signal, min_size=10))
    h2 = _hash(predict_with("numpy-pelt-l2", ds.signal, min_size=10))
    assert h1 == h2
    r1 = predict_with("numpy-mmd-rff", ds.signal, min_size=10, seed=7)
    r2 = predict_with("numpy-mmd-rff", ds.signal, min_size=10, seed=7)
    assert r1 == r2
    # RFF 换 seed 结果可以不同，但同样 seed 必须逐位一致（上面已验）


# --------------------------------------------------------------------------- I12a
@pytest.mark.slow
def test_i12a_null_fpr_gated():
    """Null-iid 档：ConStab 置换闸门的 per-series 假阳率 <= alpha + 二项波动。

    用轻配置（l2 单视图 x 3 lam）测同一闸门链路；RFF 视图行为由 I11 的
    确定性测试与基准层 Gate A 覆盖。
    """
    from changeforge.detectors import build

    alpha = 0.1
    n_series = 20
    flags = 0
    for idx in range(n_series):
        rng = np.random.default_rng(stable_seed("i12a", idx))
        x = rng.standard_normal(size=(500, 1))
        det = build(
            "constab-cpd",
            alpha=alpha,
            n_perm=19,
            seed=idx,
            min_size=10,
            feature_views=("l2",),
            lam_grid=(1.0, 3.0, 8.0),
        )
        det.fit(x)
        cps = det.predict()
        if len(cps) > 0:
            flags += 1
    # 经验 FPR 上界：alpha + 2*sqrt(alpha(1-alpha)/n)（95% 二项）
    bound = alpha + 2.0 * np.sqrt(alpha * (1 - alpha) / n_series)
    assert flags / n_series <= bound, (flags, n_series, bound)


# --------------------------------------------------------------------------- 辅助
def test_stable_seed_and_tiers():
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)
    for tier in TIER_ORDER:
        ds = make_tier_series(tier, 0)
        ds2 = make_tier_series(tier, 0)
        assert np.array_equal(ds.signal, ds2.signal)
        assert ds.n_samples == ds.n_samples


def test_circular_block_permute_shape():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, size=237)
    p = circular_block_permute(x, 7, rng)
    assert p.shape == x.shape
    # 循环置换保持 multiset（排序后逐位相等）
    assert np.array_equal(np.sort(p), np.sort(x))
