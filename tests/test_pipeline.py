"""流水线与基准测试（用假检测器打通端到端）。

作者: 晨星
"""

from __future__ import annotations

import json

import pytest

from changeforge.core.config import Config
from changeforge.core.errors import DetectorNotFoundError
from changeforge.core.types import Dataset
from changeforge.data import generate_dataset
from changeforge.detectors import build, list_detectors
from changeforge.pipeline import (
    BenchmarkRecord,
    ChangeForgePipeline,
    benchmark,
    format_table,
    save_benchmark,
)


def test_registry_empty_raises(empty_registry):
    assert list_detectors() == []
    with pytest.raises(DetectorNotFoundError) as excinfo:
        build("dummy")
    assert excinfo.value.code == "E301"


def test_dummy_detector_registered(dummy_detector_module):
    assert "dummy" in list_detectors()
    detector = build("dummy", change_points=[10, 20])
    assert detector.get_params() == {"change_points": [10, 20]}


def test_pipeline_run_with_dummy(dummy_detector_module, tiny_signal):
    signal, truth = tiny_signal
    pipeline = ChangeForgePipeline(Config(tolerance=2, min_size=5))
    result = pipeline.run(
        signal=signal,
        true_change_points=truth,
        detector="dummy",
        params={"change_points": truth},
    )
    assert result.change_points == truth
    assert result.evaluation_report is not None
    assert result.evaluation_report.f1 == pytest.approx(1.0)
    assert result.dataset_summary["n_samples"] == 100
    assert len(pipeline.history) == 1


def test_pipeline_run_synthetic(dummy_detector_module):
    pipeline = ChangeForgePipeline(Config(tolerance=5))
    dataset = generate_dataset(n_samples=300, n_change_points=2, seed=4)
    result = pipeline.run(
        dataset=dataset, detector="dummy", params={"change_points": dataset.change_points}
    )
    assert result.f1 == pytest.approx(1.0)


def test_pipeline_dataset_and_signal_conflict(dummy_detector_module, tiny_signal):
    signal, _ = tiny_signal
    dataset = Dataset(name="x", signal=signal, change_points=[30])
    with pytest.raises(Exception):
        ChangeForgePipeline().run(dataset=dataset, signal=signal, detector="dummy")


def test_pipeline_no_truth_warns(dummy_detector_module, tiny_signal):
    signal, _ = tiny_signal
    pipeline = ChangeForgePipeline(Config(tolerance=1))
    with pytest.warns(RuntimeWarning):
        result = pipeline.run(signal=signal, detector="dummy")
    assert result.evaluation_report is None


def test_pipeline_coerces_out_of_range_points(dummy_detector_module, tiny_signal):
    signal, _ = tiny_signal
    pipeline = ChangeForgePipeline(Config(tolerance=1))
    with pytest.warns(RuntimeWarning):
        result = pipeline.run(
            signal=signal,
            true_change_points=[30],
            detector="dummy",
            params={"change_points": [30, 999]},
        )
    assert result.change_points == [30]


def test_pipeline_no_detectors_registered(empty_registry, tiny_signal):
    signal, _ = tiny_signal
    with pytest.raises(DetectorNotFoundError):
        ChangeForgePipeline().run(signal=signal, detector="dummy")


def test_benchmark_and_table(dummy_detector_module):
    config = Config(tolerance=5, output_dir="artifacts")
    datasets = [generate_dataset(n_samples=300, n_change_points=2, seed=i) for i in range(2)]
    records = benchmark(datasets, ["dummy"], config=config, verbose=False)
    assert len(records) == 2
    assert all(record.status == "ok" for record in records)
    table = format_table(records)
    assert "dataset" in table.splitlines()[0]
    assert len(table.splitlines()) == 4


def test_benchmark_failure_recorded(dummy_detector_module):
    datasets = [generate_dataset(n_samples=300, n_change_points=2, seed=1)]
    records = benchmark(datasets, ["missing-detector"], config=Config(), verbose=False)
    assert records[0].status == "error"
    assert records[0].error


def test_benchmark_saves_json(tmp_path, dummy_detector_module):
    config = Config(tolerance=5, output_dir=str(tmp_path))
    datasets = [generate_dataset(n_samples=300, n_change_points=2, seed=1)]
    records = benchmark(datasets, ["dummy"], config=config, verbose=False)
    path = save_benchmark(records, config=config)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["n_records"] == 1
    assert payload["records"][0]["dataset"] == datasets[0].name
    assert payload["config"]["tolerance"] == 5


def test_benchmark_record_nan_becomes_null():
    record = BenchmarkRecord(dataset="d", detector="x")
    assert record.to_dict()["f1"] is None
