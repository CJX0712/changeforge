"""超参搜索层（Optuna）。

本层只依赖 ``core`` 与 optuna；target 目标函数由 pipeline 注入，
避免 hpo 反向依赖 detectors/eval。

作者: 晨星
"""

from __future__ import annotations

from .search import (
    HPOResult,
    SearchSpace,
    make_search_space,
    run_hpo,
)

__all__ = [
    "HPOResult",
    "SearchSpace",
    "make_search_space",
    "run_hpo",
]
