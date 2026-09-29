"""难度单调性探针：量化「采样噪声」与「容差口径」两个混淆因子。

背景
----
quick 基准里 D3/D4/D5 各只跑了 1~2 条序列，且出现了难度倒挂
（D5 Expert 的 F1 反而高于 D3/D4）。本脚本回答两个问题：

1. 倒挂是采样噪声吗？→ 每档跑 N 条序列，给 mean ± std。
2. 容差口径可比吗？→ 同时用两种 tol：
     tol_rel   = 0.01 * n          （现行口径，随 n 放大）
     tol_fixed = 固定值             （跨档可比）
   并报告 tol / mean_seg_len —— 该比值越大，命中越容易。

只有跑得快、不拖累整体基准的检测器入选（跳过 constab 与慢速 ruptures）。

用法:
    python scripts/difficulty_probe.py [--n 20] [--tol-fixed 10]

所有数字来自真实重跑，不复用旧结果。作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TIERS = ["D0", "D3", "D4", "D5"]
FAST_DETECTORS = [
    "numpy-pelt-l2",
    "numpy-binseg-l2",
    "numpy-window-l2",
    "numpy-cusum",
    "numpy-slidingt",
    "ruptures-kernel-rbf",
]


def _probe(n_series: int, tol_fixed: int) -> dict:
    import sys

    sys.path.insert(0, str(ROOT))
    import changeforge  # noqa: F401
    from changeforge.core.errors import InvalidDetectorParamsError
    from changeforge.data.tiers import TIER_SPECS, make_tier_series
    from changeforge.detectors import build
    from changeforge.eval.metrics import evaluate

    out: dict = {}
    for tier in TIERS:
        spec = TIER_SPECS[tier]
        rec: dict = {
            "tier": tier,
            "n": spec.n,
            "k": spec.k,
            "n_series": n_series,
            "detectors": {},
        }
        seg_lens: list[float] = []
        for name in FAST_DETECTORS:
            f1_rel: list[float] = []
            f1_fix: list[float] = []
            f1_norm: list[float] = []
            for idx in range(n_series):
                ds = make_tier_series(tier, idx)
                sig = ds.signal
                n = ds.n_samples
                true = ds.change_points
                ms = max(2, round(0.01 * n))
                if true:
                    bounds = [0, *true, n]
                    seg_lens.append(float(np.mean(np.diff(bounds))))
                try:
                    det = build(name, min_size=ms)
                except InvalidDetectorParamsError:
                    det = build(name)
                det.fit(sig)
                cps = [int(c) for c in det.predict()]
                # 口径 C：按真值平均段长归一 -> tol/seg 恒为 0.1，跨档可比
                tol_norm = max(1, round(0.1 * n / (len(true) + 1)))
                f1_rel.append(evaluate(true, cps, n, tolerance=max(1, round(0.01 * n))).f1)
                f1_fix.append(evaluate(true, cps, n, tolerance=tol_fixed).f1)
                f1_norm.append(evaluate(true, cps, n, tolerance=tol_norm).f1)
            rec["detectors"][name] = {
                "f1_tol_rel_mean": round(float(np.mean(f1_rel)), 4),
                "f1_tol_rel_std": round(float(np.std(f1_rel)), 4),
                "f1_tol_fixed_mean": round(float(np.mean(f1_fix)), 4),
                "f1_tol_fixed_std": round(float(np.std(f1_fix)), 4),
                "f1_tol_norm_mean": round(float(np.mean(f1_norm)), 4),
                "f1_tol_norm_std": round(float(np.std(f1_norm)), 4),
            }
        mean_seg = float(np.mean(seg_lens)) if seg_lens else float("nan")
        rec["mean_seg_len"] = round(mean_seg, 1)
        rec["tol_rel"] = max(1, round(0.01 * spec.n))
        rec["tol_fixed"] = tol_fixed
        rec["tol_norm"] = max(1, round(0.1 * spec.n / (spec.k + 1)))
        rec["tol_norm_over_seg"] = (
            round(rec["tol_norm"] / mean_seg, 3) if mean_seg == mean_seg else None
        )
        # 相对容差：越大越容易命中（可比性的关键混淆因子）
        rec["tol_rel_over_seg"] = (
            round(rec["tol_rel"] / mean_seg, 3) if mean_seg == mean_seg else None
        )
        rec["tol_fixed_over_seg"] = (
            round(tol_fixed / mean_seg, 3) if mean_seg == mean_seg else None
        )
        # 跨检测器的档位平均分（口径可比的核心指标）
        rec["macro_tol_rel"] = round(
            float(np.mean([v["f1_tol_rel_mean"] for v in rec["detectors"].values()])), 4
        )
        rec["macro_tol_fixed"] = round(
            float(np.mean([v["f1_tol_fixed_mean"] for v in rec["detectors"].values()])), 4
        )
        rec["macro_tol_norm"] = round(
            float(np.mean([v["f1_tol_norm_mean"] for v in rec["detectors"].values()])), 4
        )
        out[tier] = rec
        print(
            f"[{tier}] rel={rec['macro_tol_rel']:.3f} fix={rec['macro_tol_fixed']:.3f} "
            f"norm={rec['macro_tol_norm']:.3f} | mean_seg={mean_seg:.0f} "
            f"tol/seg: rel={rec['tol_rel_over_seg']:.2f} "
            f"fix={rec['tol_fixed_over_seg']:.2f} "
            f"norm={rec['tol_norm_over_seg']:.2f}",
            flush=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每档序列数")
    ap.add_argument("--tol-fixed", type=int, default=10, help="固定容差口径")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "difficulty_probe.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    res = _probe(args.n, args.tol_fixed)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "author": "晨星",
        "n_series_per_tier": args.n,
        "tol_fixed": args.tol_fixed,
        "detectors": FAST_DETECTORS,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "tiers": res,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote {out}  ({report['elapsed_s']}s)")
    print(
        f"\n{'tier':<5}{'K':>4}{'mean_seg':>10}{'tol':>6}{'tol/seg':>9}"
        f"{'@rel':>9}{'@fix':>9}{'@norm':>9}"
    )
    for tier in TIERS:
        r = res[tier]
        print(
            f"{tier:<5}{r['k']:>4}{r['mean_seg_len']:>10.0f}"
            f"{r['tol_norm']:>6}"
            f"{r['tol_norm_over_seg'] or 0:>9.2f}"
            f"{r['macro_tol_rel']:>9.3f}{r['macro_tol_fixed']:>9.3f}"
            f"{r['macro_tol_norm']:>9.3f}"
        )

    # 单调性判定（按 K 递增 = 难度递增的名义顺序 D0 < D3 < D4 < D5）
    order = ["D0", "D3", "D4", "D5"]
    rel = [res[t]["macro_tol_rel"] for t in order]
    fix = [res[t]["macro_tol_fixed"] for t in order]
    nor = [res[t]["macro_tol_norm"] for t in order]

    def mono(v: list[float]) -> bool:
        return all(v[i] >= v[i + 1] for i in range(len(v) - 1))

    print("\n难度单调性（名义难度 D0 < D3 < D4 < D5，故 F1 应递减）:")
    print(f"  口径 A tol=0.01n      : {'单调 OK' if mono(rel) else '非单调'}  {rel}")
    print(
        f"  口径 B tol={args.tol_fixed} 固定   : {'单调 OK' if mono(fix) else '非单调'}  {fix}"
    )
    print(f"  口径 C tol=0.1*seg 归一: {'单调 OK' if mono(nor) else '非单调'}  {nor}")


if __name__ == "__main__":
    main()
