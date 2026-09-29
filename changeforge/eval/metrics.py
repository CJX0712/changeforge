"""变化点检测评测指标。

核心是**容差匹配**：预测点 ``p`` 与真值点 ``t`` 在 ``|p - t| <= tolerance`` 内
且一对一配对时算命中（TP）。这是变化点检测社区（如 TCPD / SMD 基准）的通用口径。

指标：
    precision / recall / f1  容差匹配下的分类指标
    matched                  TP 数量
    mean_abs_error           已匹配点对的绝对索引误差均值
    count_error              n_pred - n_true（正数=过分割）
    adjusted_rand_index      由变化点还原分段标签后的 ARI

作者: 晨星
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..core.errors import EvaluationError, InvalidChangePointsError
from ..core.types import EvalReport

__all__ = [
    "adjusted_rand_index",
    "evaluate",
    "evaluate_many",
    "labels_from_change_points",
    "match_change_points",
]


def _as_sorted_unique(points: Sequence[int] | np.ndarray | None, n_samples: int, tag: str):
    if points is None:
        return []
    raw: list[int] = []
    for item in list(points):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise InvalidChangePointsError("变化点必须是整数", name=tag, value=item)
        value = int(item)
        if not 1 <= value <= n_samples - 1:
            raise InvalidChangePointsError(
                "变化点越界", name=tag, value=value, n_samples=n_samples
            )
        raw.append(value)
    return sorted(set(raw))


def match_change_points(
    true_points: Sequence[int],
    pred_points: Sequence[int],
    tolerance: int = 0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """贪心一对一容差匹配。

    按预测点顺序，为每个预测点挑选**最近的**未匹配真值点（距离 <= tolerance）。

    Returns:
        (matched_pairs, unmatched_true, unmatched_pred)
        matched_pairs 元素为 (true, pred)。
    """
    if tolerance < 0:
        raise EvaluationError("tolerance 不能为负数", tolerance=tolerance)
    remaining_true = sorted(set(int(t) for t in true_points))
    preds = sorted(set(int(p) for p in pred_points))
    pairs: list[tuple[int, int]] = []
    for pred in preds:
        best_idx = -1
        best_dist = None
        for idx, truth in enumerate(remaining_true):
            dist = abs(truth - pred)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_idx, best_dist = idx, dist
        if best_idx >= 0:
            pairs.append((int(remaining_true[best_idx]), pred))
            remaining_true.pop(best_idx)
    matched_true = {truth for truth, _ in pairs}
    matched_pred = {pred for _, pred in pairs}
    unmatched_true = [
        t for t in sorted(set(int(t) for t in true_points)) if t not in matched_true
    ]
    unmatched_pred = [p for p in preds if p not in matched_pred]
    return pairs, unmatched_true, unmatched_pred


def labels_from_change_points(change_points: Sequence[int], n_samples: int) -> np.ndarray:
    """把变化点还原为逐点分段标签（用于 ARI 等划分指标）。"""
    labels = np.zeros(n_samples, dtype=np.int64)
    cursor = 0
    for seg_id, point in enumerate(sorted(set(int(c) for c in change_points)), start=1):
        point = int(np.clip(point, 1, n_samples - 1))
        labels[cursor:point] = seg_id - 1
        cursor = point
    labels[cursor:] = len(change_points)
    return labels


def _comb2(value: float) -> float:
    return value * (value - 1.0) / 2.0


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """调整兰德指数（自行实现，避免引入 sklearn 依赖）。"""
    a = np.asarray(labels_true, dtype=np.int64).reshape(-1)
    b = np.asarray(labels_pred, dtype=np.int64).reshape(-1)
    if a.shape != b.shape:
        raise EvaluationError("标签长度不一致", a=a.shape, b=b.shape)
    n = int(a.shape[0])
    if n < 2:
        return 1.0
    codes_true, true_idx = np.unique(a, return_inverse=True)
    codes_pred, pred_idx = np.unique(b, return_inverse=True)
    contingency = np.zeros((codes_true.size, codes_pred.size), dtype=np.float64)
    np.add.at(contingency, (true_idx.reshape(-1), pred_idx.reshape(-1)), 1.0)
    sum_comb = float(np.sum(_comb2(contingency)))
    sum_a = float(np.sum(_comb2(np.bincount(true_idx).astype(np.float64))))
    sum_b = float(np.sum(_comb2(np.bincount(pred_idx).astype(np.float64))))
    total = _comb2(float(n))
    expected = sum_a * sum_b / total if total > 0 else 0.0
    denominator = 0.5 * (sum_a + sum_b) - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return 1.0
    return float((sum_comb - expected) / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def evaluate(
    true_points: Sequence[int] | None,
    pred_points: Sequence[int] | None,
    n_samples: int,
    tolerance: int = 0,
) -> EvalReport:
    """评测一组预测变化点。

    Args:
        true_points: 真值变化点（可为空）。
        pred_points: 预测变化点（可为空）。
        n_samples: 信号长度（用于边界校验与标签还原）。
        tolerance: 匹配容差（样本点数）。

    Returns:
        EvalReport。

    Raises:
        EvaluationError: n_samples 非法或 tolerance 为负。
        InvalidChangePointsError: 变化点越界或非整数。
    """
    if n_samples is None or n_samples < 1:
        raise EvaluationError("n_samples 非法", n_samples=n_samples)
    if tolerance < 0:
        raise EvaluationError("tolerance 不能为负数", tolerance=tolerance)

    truth = _as_sorted_unique(true_points, n_samples, "true_points")
    preds = _as_sorted_unique(pred_points, n_samples, "pred_points")

    pairs, _, _ = match_change_points(truth, preds, tolerance)
    matched = len(pairs)
    precision = _safe_ratio(matched, len(preds)) if preds else (1.0 if not truth else 0.0)
    recall = _safe_ratio(matched, len(truth)) if truth else (1.0 if not preds else 0.0)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    if pairs:
        mean_abs_error = float(np.mean([abs(t - p) for t, p in pairs]))
    else:
        mean_abs_error = float("nan")
    ari = adjusted_rand_index(
        labels_from_change_points(truth, n_samples),
        labels_from_change_points(preds, n_samples),
    )
    return EvalReport(
        n_true=len(truth),
        n_pred=len(preds),
        matched=matched,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_abs_error=float(mean_abs_error),
        count_error=len(preds) - len(truth),
        adjusted_rand_index=float(ari),
        tolerance=int(tolerance),
    )


def evaluate_many(
    true_points: Sequence[int] | None,
    pred_sets: Sequence[Sequence[int]],
    n_samples: int,
    tolerance: int = 0,
) -> list[EvalReport]:
    """对同一真值批量评测多组预测（基准测试用）。"""
    return [
        evaluate(true_points, preds, n_samples=n_samples, tolerance=tolerance)
        for preds in pred_sets
    ]
