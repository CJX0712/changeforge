"""数据层：合成数据生成 + 载入。

依赖方向：只依赖 ``core``。
"""

from __future__ import annotations

from .loader import (
    from_array,
    load_csv,
    load_dataset,
    load_npy,
    write_dataset_csv,
)
from .synthetic import (
    SYNTHETIC_KINDS,
    generate_dataset,
    make_synthetic_suite,
)

__all__ = [
    "SYNTHETIC_KINDS",
    "from_array",
    "generate_dataset",
    "load_csv",
    "load_dataset",
    "load_npy",
    "make_synthetic_suite",
    "write_dataset_csv",
]
