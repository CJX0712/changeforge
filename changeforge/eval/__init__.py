"""评测层：变化点检测指标。

只依赖 ``core`` 与 numpy，不引入 sklearn，规避"评测函数与 sklearn 导入别名
同名导致递归导入"的历史坑。

作者: 晨星
"""

from __future__ import annotations

from .metrics import (
    adjusted_rand_index,
    evaluate,
    evaluate_many,
    labels_from_change_points,
    match_change_points,
)

__all__ = [
    "adjusted_rand_index",
    "evaluate",
    "evaluate_many",
    "labels_from_change_points",
    "match_change_points",
]
