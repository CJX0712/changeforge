"""ChangeForge 命令行入口（argparse）。

子命令::

    info              打印版本 / 配置 / 环境
    list-detectors    列出已注册检测器
    list-datasets     列出合成数据类型
    gen               生成合成数据并导出 CSV
    run               单数据集端到端：检测 + 评测
    benchmark         多数据集 x 多检测器基准，落盘 benchmark.json
    tune              Optuna 调参（需 --space 指定搜索空间 JSON）

用法::

    python -m changeforge.cli info
    python -m changeforge.cli benchmark --output-dir artifacts

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from . import __version__
from .core.config import config_from_file, load_config
from .core.errors import ChangeForgeError, CLIError
from .core.types import Dataset
from .data import SYNTHETIC_KINDS, generate_dataset, load_dataset, make_synthetic_suite
from .data.loader import write_dataset_csv
from .detectors import get_registry_doc, list_detectors
from .pipeline import ChangeForgePipeline, benchmark, format_table, save_benchmark

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="changeforge",
        description="ChangeForge —— 变化点检测（Changepoint Detection）工程框架",
    )
    parser.add_argument("--version", action="version", version=f"changeforge {__version__}")
    parser.add_argument("--config", help="JSON 配置文件路径（覆盖默认值与环境变量）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--tolerance", type=int, default=None, help="评测匹配容差（样本点）")
    parser.add_argument("--min-size", type=int, default=None, help="最小段长度")
    parser.add_argument("--output-dir", default=None, help="产物输出目录")
    parser.add_argument(
        "--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--detector", default=None, help="默认检测器名")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="打印版本 / 配置 / 环境")
    subparsers.add_parser("list-detectors", help="列出已注册检测器")
    subparsers.add_parser("list-datasets", help="列出合成数据类型")

    gen = subparsers.add_parser("gen", help="生成合成数据并导出 CSV")
    gen.add_argument("--kind", default="constant", choices=list(SYNTHETIC_KINDS))
    gen.add_argument("--n-samples", type=int, default=1000)
    gen.add_argument("--n-change-points", type=int, default=3)
    gen.add_argument("--noise-std", type=float, default=0.3)
    gen.add_argument("--out", default=None, help="输出 CSV 路径")

    run = subparsers.add_parser("run", help="单数据集端到端：检测 + 评测")
    run.add_argument("--kind", default="constant", choices=list(SYNTHETIC_KINDS))
    run.add_argument("--n-samples", type=int, default=1000)
    run.add_argument("--n-change-points", type=int, default=3)
    run.add_argument("--input", default=None, help="从 CSV/NPY 载入（优先于合成数据）")
    run.add_argument("--detector", default=None)
    run.add_argument("--json", action="store_true", help="以 JSON 输出结果")

    bench = subparsers.add_parser("benchmark", help="多数据集 x 多检测器基准")
    bench.add_argument("--n-samples", type=int, default=1000)
    bench.add_argument("--n-change-points", type=int, default=3)
    bench.add_argument("--kinds", default=None, help="逗号分隔的合成类型，默认全部")
    bench.add_argument("--detectors", default=None, help="逗号分隔的检测器名，默认全部已注册")
    bench.add_argument("--out", default=None, help="benchmark.json 输出路径")

    tune = subparsers.add_parser("tune", help="Optuna 调参")
    tune.add_argument("--detector", required=True)
    tune.add_argument("--space", required=True, help="搜索空间 JSON 文件路径")
    tune.add_argument("--kind", default="constant", choices=list(SYNTHETIC_KINDS))
    tune.add_argument("--n-trials", type=int, default=None)
    tune.add_argument("--timeout", type=float, default=None)
    tune.add_argument("--metric", default=None, help="优化指标，默认 f1")
    return parser


def _resolve_config(args: argparse.Namespace):
    overrides: dict[str, Any] = {
        "seed": args.seed,
        "tolerance": args.tolerance,
        "min_size": args.min_size,
        "output_dir": args.output_dir,
        "log_level": args.log_level,
    }
    if getattr(args, "detector", None):
        overrides["detector"] = args.detector
    if args.config:
        return config_from_file(args.config).with_overrides(**overrides)
    return load_config(**overrides)


def _require_detectors(config) -> list[str]:
    names = list_detectors()
    if not names:
        raise CLIError(
            "检测器注册表为空：算法模块尚未接入（Phase 2b）。 当前只能跑数据生成与评测自检。",
            command="detectors",
        )
    if config.detector != "auto" and config.detector not in names:
        raise CLIError("配置的默认检测器未注册", detector=config.detector, available=names)
    return names


def _make_dataset(args: argparse.Namespace, config) -> Dataset:
    if getattr(args, "input", None):
        return load_dataset(args.input)
    return generate_dataset(
        kind=args.kind,
        n_samples=args.n_samples,
        n_change_points=args.n_change_points,
        seed=config.seed,
    )


def _cmd_info(config) -> int:
    payload = {
        "version": __version__,
        "author": "晨星",
        "python": sys.version.split()[0],
        "config": config.to_dict(),
        "detectors": list_detectors(),
        "synthetic_kinds": list(SYNTHETIC_KINDS),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return EXIT_OK


def _cmd_list_detectors() -> int:
    names = list_detectors()
    if not names:
        print("(注册表为空：算法模块尚未接入，Phase 2b 落地后此处会列出检测器)")
        return EXIT_OK
    print(f"{'name':<24}{'description'}")
    for name in names:
        print(f"{name:<24}{get_registry_doc(name)}")
    return EXIT_OK


def _cmd_list_datasets() -> int:
    print(f"{'kind':<16}{'description'}")
    descriptions = {
        "constant": "分段常数 + 噪声（均值漂移）",
        "variance": "分段方差切换（均值恒定）",
        "trend": "分段线性趋势，斜率切换",
        "mixed": "均值 + 方差同时切换",
    }
    for kind in SYNTHETIC_KINDS:
        print(f"{kind:<16}{descriptions.get(kind, '')}")
    return EXIT_OK


def _cmd_gen(args: argparse.Namespace, config) -> int:
    dataset = generate_dataset(
        kind=args.kind,
        n_samples=args.n_samples,
        n_change_points=args.n_change_points,
        noise_std=args.noise_std,
        seed=config.seed,
    )
    target = args.out or f"{config.output_dir}/{dataset.name}.csv"
    path = write_dataset_csv(dataset, target)
    print(f"已生成: {dataset.name} n={dataset.n_samples} cps={dataset.change_points}")
    print(f"CSV: {path}")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace, config) -> int:
    names = _require_detectors(config)
    detector = args.detector or (config.detector if config.detector != "auto" else names[0])
    dataset = _make_dataset(args, config)
    result = ChangeForgePipeline(config).run(
        dataset=dataset, detector=detector, tolerance=config.tolerance
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
        return EXIT_OK
    print(f"dataset   : {result.dataset_name} (n={result.dataset_summary['n_samples']})")
    print(f"detector  : {result.detector_name} params={result.params}")
    print(f"truth     : {dataset.change_points}")
    print(f"predicted : {result.change_points}")
    print(f"elapsed   : {result.elapsed_ms:.2f} ms")
    if result.evaluation_report is not None:
        print(json.dumps(result.evaluation_report.to_dict(), ensure_ascii=False, indent=2))
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace, config) -> int:
    detectors = (
        [item.strip() for item in args.detectors.split(",") if item.strip()]
        if args.detectors
        else _require_detectors(config)
    )
    kinds = (
        tuple(item.strip() for item in args.kinds.split(",") if item.strip())
        if args.kinds
        else SYNTHETIC_KINDS
    )
    datasets = make_synthetic_suite(
        seed=config.seed,
        n_samples=args.n_samples,
        kinds=tuple(kinds),
        n_change_points=args.n_change_points,
    )
    records = benchmark(datasets, detectors, config=config, verbose=True)
    print()
    print(format_table(records))
    path = save_benchmark(records, path=args.out, config=config)
    print(f"\nbenchmark.json -> {path}")
    return EXIT_OK


def _cmd_tune(args: argparse.Namespace, config) -> int:
    _require_detectors(config)
    with open(args.space, encoding="utf-8") as handle:
        try:
            space = json.load(handle)
        except json.JSONDecodeError as exc:
            raise CLIError("搜索空间 JSON 解析失败", path=args.space) from exc
    dataset = generate_dataset(
        kind=args.kind,
        n_samples=1000,
        n_change_points=3,
        seed=config.seed,
    )
    result = ChangeForgePipeline(config).tune(
        dataset=dataset,
        detector=args.detector,
        search_space_dict=space,
        metric=args.metric,
        n_trials=args.n_trials,
        timeout_s=args.timeout,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _resolve_config(args)
    except ChangeForgeError as exc:
        print(exc.render(), file=sys.stderr)
        return EXIT_USAGE
    handlers = {
        "info": lambda: _cmd_info(config),
        "list-detectors": _cmd_list_detectors,
        "list-datasets": _cmd_list_datasets,
        "gen": lambda: _cmd_gen(args, config),
        "run": lambda: _cmd_run(args, config),
        "benchmark": lambda: _cmd_benchmark(args, config),
        "tune": lambda: _cmd_tune(args, config),
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse 已保证
        print(f"未知子命令: {args.command}", file=sys.stderr)
        return EXIT_USAGE
    try:
        return int(handler())
    except ChangeForgeError as exc:
        print(exc.render(), file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"[E203] 文件不存在: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
