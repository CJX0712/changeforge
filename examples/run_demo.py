"""ChangeForge 五档基准（真实数字落盘 benchmark.json + dgp_calibration.json）。

口径（与 docs/architecture.md 一致）:
    - 主指标 F1@tol_rel, tol_rel = 0.01 * n
    - Null 档报告 per-series 假阳率（检出 >= 1 个变点的序列占比）
    - default 协议（lam=3 归一化 BIC 罚，全检测器同口径）；oracle-K 单列禁入主表
    - runtime: time.perf_counter 单次实测（多序列取均值）
    - Gate A/B/C 审计信息随结果落盘（不隐藏任何数字）

跑法::

    python examples/run_demo.py            # 常规规模
    python examples/run_demo.py --quick    # CI 冒烟

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import changeforge  # noqa: E402
from changeforge.core.errors import InvalidDetectorParamsError  # noqa: E402
from changeforge.data.tiers import TIER_ORDER, TIER_SPECS, make_tier_series  # noqa: E402
from changeforge.detectors import build  # noqa: E402
from changeforge.eval.metrics import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MAIN_DETECTORS = [
    "numpy-pelt-l2",
    "numpy-binseg-l2",
    "numpy-window-l2",
    "numpy-mmd-rff",
    "numpy-cusum",
    "numpy-slidingt",
    "ruptures-pelt-l2",
    "ruptures-pelt-l1",
    "ruptures-pelt-normal",
    "ruptures-pelt-rank",
    "ruptures-kernel-rbf",
    "constab-cpd",
]
ORACLE = "ruptures-dynp-l2"  # 仅 oracle-K 参考，禁止进主表
# 仅这两个检测器接受 seed 作为构造参数；其余传 seed 会触发 E303 被跳过
_SEED_AWARE = {"constab-cpd", "numpy-mmd-rff"}

SERIES_PER_TIER = {"D0": 5, "D1": 24, "D1b": 24, "D3": 8, "D4": 4, "D5": 2}
QUICK_PER_TIER = {"D0": 2, "D1": 5, "D1b": 5, "D3": 2, "D4": 1, "D5": 1}


def _tol(n: int) -> int:
    """口径 A：容差 = 1% 序列长度（行业常见，但与变点数耦合，跨档不可比）。"""
    return max(1, round(0.01 * n))


def _tol_norm(n: int, k_true: int) -> int:
    """口径 C：容差 = 10% 平均段长（tol/seg 恒为 0.1，跨档可比）。

    口径 A 下 tol/seg = 0.01*K，D5(K=20) 达 0.21 而 D0 仅 0.02，
    相差 10 倍——密集变点档会被容差"送分"，造成难度倒挂假象。
    本口径消除该混淆因子（代价：用到了真值 K 的弱先验，仅作评测口径）。
    """
    return max(1, round(0.1 * n / (k_true + 1)))


def _atomic_write(path: Path, data: dict) -> Path:
    """原子写：先写临时文件再 rename；若目标被残留进程锁住，回退到 .out.json。"""
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        return path
    except (PermissionError, OSError):
        fallback = path.with_name(path.stem + ".out.json")
        fallback.write_text(payload, encoding="utf-8")
        print(f"[warn] {path.name} 被锁，已回退写入 {fallback.name}", flush=True)
        return fallback


def run(quick: bool = False) -> dict:
    per_tier = QUICK_PER_TIER if quick else SERIES_PER_TIER
    partial_path = ROOT / "artifacts" / ".bench.partial.json"
    partial: dict = {}
    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
        except Exception:
            partial = {}
    results = partial.get("results", {tier: {} for tier in TIER_ORDER})
    for tier in TIER_ORDER:
        results.setdefault(tier, {})
    oracle_ref = partial.get("oracle", {})
    done_tiers = set(partial.get("done_tiers", []))
    started = time.perf_counter()
    for tier in TIER_ORDER:
        if tier in done_tiers:
            print(f"[{tier}] resume-skip (已完成)", flush=True)
            continue
        spec = TIER_SPECS[tier]
        count = per_tier[tier]
        for det_name in MAIN_DETECTORS:
            f1s, prs, rcs, els, flagged, gate_logs = [], [], [], [], 0, []
            f1s_norm, prs_norm, rcs_norm = [], [], []
            for idx in range(count):
                ds = make_tier_series(tier, idx)
                n = ds.n_samples
                ms = max(2, round(0.01 * n))
                try:
                    extra = {"seed": idx} if det_name in _SEED_AWARE else {}
                    try:
                        det = build(det_name, min_size=ms, **extra)
                    except InvalidDetectorParamsError:
                        # 个别检测器（如 numpy-cusum）不接受 min_size，退回不带该参数构造
                        det = build(det_name, **extra)
                    det.fit(ds.signal)
                    t0 = time.perf_counter()
                    cps = [int(c) for c in det.predict()]
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    gate = det.last_gate_metrics() if hasattr(det, "last_gate_metrics") else {}
                except Exception as exc:  # 诚实记录失败而非中断
                    gate_logs.append({"error": f"{type(exc).__name__}: {exc}"})
                    continue
                rep = evaluate(ds.change_points, cps, n, tolerance=_tol(n))
                rep_n = evaluate(
                    ds.change_points,
                    cps,
                    n,
                    tolerance=_tol_norm(n, len(ds.change_points)),
                )
                f1s.append(rep.f1)
                prs.append(rep.precision)
                rcs.append(rep.recall)
                f1s_norm.append(rep_n.f1)
                prs_norm.append(rep_n.precision)
                rcs_norm.append(rep_n.recall)
                els.append(elapsed)
                if cps:
                    flagged += 1
                if gate:
                    gate_logs.append(gate)
            if not f1s:
                continue
            entry = {
                "f1_mean": round(float(np.mean(f1s)), 4),
                "precision_mean": round(float(np.mean(prs)), 4),
                "recall_mean": round(float(np.mean(rcs)), 4),
                "f1_mean_norm": round(float(np.mean(f1s_norm)), 4),
                "precision_mean_norm": round(float(np.mean(prs_norm)), 4),
                "recall_mean_norm": round(float(np.mean(rcs_norm)), 4),
                "runtime_ms_mean": round(float(np.mean(els)), 1),
                "n_series": len(f1s),
            }
            if spec.is_null:
                entry["fpr_per_series"] = round(flagged / count, 4)
            if gate_logs:
                entry["gate_sample"] = gate_logs[-1]
            results[tier][det_name] = entry
        if not spec.is_null:
            of1, of1_norm = [], []
            for idx in range(min(count, 2)):
                ds = make_tier_series(tier, idx)
                n = ds.n_samples
                ms = max(2, round(0.01 * n))
                det = build(ORACLE, n_bkps=spec.k, min_size=ms)
                det.fit(ds.signal)
                cps = [int(c) for c in det.predict()]
                rep = evaluate(ds.change_points, cps, n, tolerance=_tol(n))
                rep_n = evaluate(
                    ds.change_points,
                    cps,
                    n,
                    tolerance=_tol_norm(n, len(ds.change_points)),
                )
                of1.append(rep.f1)
                of1_norm.append(rep_n.f1)
            if of1:
                oracle_ref[tier] = {
                    "detector": ORACLE,
                    "k_given": spec.k,
                    "f1_mean": round(float(np.mean(of1)), 4),
                    "f1_mean_norm": round(float(np.mean(of1_norm)), 4),
                    "note": "oracle-K 上界参考，非公平对比，禁止进主表",
                }
        done_tiers.add(tier)
        # 增量落盘（断点续跑，跨会话重启不丢进度）
        _atomic_write(
            partial_path,
            {
                "results": results,
                "oracle": oracle_ref,
                "done_tiers": sorted(done_tiers),
                "quick": quick,
            },
        )
        print(f"[{tier}] done, cumulative {time.perf_counter() - started:.0f}s", flush=True)

    # ---- 区分度自检协议（dgp_calibration） ----
    calibration = {}
    for tier in TIER_ORDER:
        spec = TIER_SPECS[tier]
        det_f1 = {k: v["f1_mean"] for k, v in results[tier].items()}
        singles = {k: v for k, v in det_f1.items() if k != "constab-cpd"}
        entry: dict = {"tier": tier, "n_series": per_tier[tier], "checks": {}}
        if spec.is_null:
            fprs = {k: v.get("fpr_per_series") for k, v in results[tier].items()}
            entry["checks"]["null_fpr"] = {k: v for k, v in fprs.items() if v is not None}
        else:
            if singles:
                entry["checks"]["best_single"] = max(singles.values())
                entry["checks"]["spread"] = round(
                    max(singles.values()) - min(singles.values()), 4
                )
            if tier == "D0" and det_f1:
                entry["checks"]["smoke_min_f1"] = min(det_f1.values())
        if "constab-cpd" in det_f1 and singles:
            gain = round(det_f1["constab-cpd"] - max(singles.values()), 4)
            entry["checks"]["fused_gain_vs_best_single"] = gain
            worst = -max(0.02, 0.03 * max(singles.values()))
            entry["checks"]["gate_b_fallback"] = bool(gain < worst)
        calibration[tier] = entry

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "author": "晨星",
        "version": changeforge.__version__,
        "mode": "quick" if quick else "full",
        "tolerance_rule": (
            "口径A: tol=0.01*n（行业常见，但与 K 耦合）; "
            "口径C: tol=0.1*n/(K_true+1)（tol/段长恒 0.1，跨档可比）"
        ),
        "penalty_rule": "beta=lam*d*sigma_hat^2*log n; sigma_hat=MAD*1.4826; lam=3(BIC-like)",
        "oracle": oracle_ref,
        "results": results,
        "dgp_calibration": calibration,
    }
    out = ROOT / "artifacts" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    final_path = _atomic_write(out, report)
    _atomic_write(ROOT / "artifacts" / "dgp_calibration.json", calibration)
    # 清理增量文件
    try:
        partial_path.unlink(missing_ok=True)
    except OSError:
        pass
    print(f"\nwrote {final_path}")
    print(f"{'tier':<5}{'detector':<20}{'F1':>8}{'P':>8}{'R':>8}{'ms':>9}  notes")
    for tier in TIER_ORDER:
        for det_name, v in results[tier].items():
            notes = [f"FPR={v['fpr_per_series']:.2f}"] if "fpr_per_series" in v else []
            print(
                f"{tier:<5}{det_name:<20}{v['f1_mean']:>8.3f}{v['precision_mean']:>8.3f}"
                f"{v['recall_mean']:>8.3f}{v['runtime_ms_mean']:>9.1f}  {' '.join(notes)}"
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeForge benchmark demo")
    parser.add_argument("--quick", action="store_true", help="CI 冒烟规模")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
