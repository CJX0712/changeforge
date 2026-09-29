"""合成数据生成测试。

作者: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from changeforge.core.errors import InvalidChangePointsError
from changeforge.data import SYNTHETIC_KINDS, generate_dataset, make_synthetic_suite
from changeforge.data.loader import load_csv, load_dataset, write_dataset_csv


@pytest.mark.parametrize("kind", SYNTHETIC_KINDS)
def test_generate_all_kinds(kind):
    dataset = generate_dataset(kind=kind, n_samples=600, n_change_points=3, seed=1)
    assert dataset.signal.shape == (600,)
    assert len(dataset.change_points) == 3
    assert dataset.change_points == sorted(set(dataset.change_points))
    assert all(1 <= c <= 599 for c in dataset.change_points)
    assert np.all(np.isfinite(dataset.signal))


def test_reproducible_with_same_seed():
    first = generate_dataset(n_samples=400, n_change_points=2, seed=11)
    second = generate_dataset(n_samples=400, n_change_points=2, seed=11)
    np.testing.assert_allclose(first.signal, second.signal)
    assert first.change_points == second.change_points


def test_different_seed_changes_output():
    first = generate_dataset(n_samples=400, n_change_points=2, seed=11)
    second = generate_dataset(n_samples=400, n_change_points=2, seed=12)
    assert not np.allclose(first.signal, second.signal)


def test_mean_shift_is_visible():
    dataset = generate_dataset(kind="constant", n_samples=600, n_change_points=2, seed=3)
    cps = [0, *dataset.change_points, 600]
    means = [dataset.signal[a:b].mean() for a, b in zip(cps[:-1], cps[1:])]
    assert max(means) - min(means) > 1.0


def test_min_size_respected():
    dataset = generate_dataset(n_samples=300, n_change_points=2, min_size=50, seed=5)
    bounds = [0, *dataset.change_points, 300]
    lengths = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
    assert min(lengths) >= 50


def test_too_many_change_points_raises():
    with pytest.raises(InvalidChangePointsError):
        generate_dataset(n_samples=100, n_change_points=20, min_size=30, seed=1)


def test_unknown_kind_raises():
    with pytest.raises(InvalidChangePointsError):
        generate_dataset(kind="no-such-kind", seed=1)


def test_zero_change_points():
    dataset = generate_dataset(n_samples=200, n_change_points=0, seed=2)
    assert dataset.change_points == []


def test_suite_has_all_kinds():
    suite = make_synthetic_suite(seed=42, n_samples=500)
    assert len(suite) == len(SYNTHETIC_KINDS)
    assert {dataset.metadata["kind"] for dataset in suite} == set(SYNTHETIC_KINDS)


def test_csv_roundtrip(tmp_path):
    dataset = generate_dataset(n_samples=120, n_change_points=2, seed=7)
    path = str(tmp_path / "signal.csv")
    write_dataset_csv(dataset, path)
    loaded = load_csv(path)
    assert loaded.n_samples == dataset.n_samples
    assert loaded.change_points == dataset.change_points
    np.testing.assert_allclose(loaded.signal, dataset.signal, atol=1e-8)


def test_load_dataset_dispatches_by_suffix(tmp_path):
    dataset = generate_dataset(n_samples=80, n_change_points=1, seed=7)
    path = str(tmp_path / "signal.csv")
    write_dataset_csv(dataset, path)
    assert load_dataset(path).n_samples == 80
    with pytest.raises(Exception):
        load_dataset(str(tmp_path / "signal.txt"))
