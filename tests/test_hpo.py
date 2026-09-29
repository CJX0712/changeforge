"""HPO 基建测试（Optuna）。

作者: 晨星
"""

from __future__ import annotations

import pytest

from changeforge.core.config import Config
from changeforge.core.errors import InvalidSearchSpaceError, StudyFailedError
from changeforge.hpo import make_search_space, run_hpo
from changeforge.hpo.search import ParamSpec, SearchSpace


def test_search_space_suggest():
    space = make_search_space(
        {
            "window": {"kind": "int", "low": 5, "high": 50, "default": 20},
            "threshold": {"kind": "float", "low": 0.0, "high": 1.0},
            "scale": {"kind": "loguniform", "low": 1e-3, "high": 1.0},
            "kernel": {"kind": "categorical", "choices": ["linear", "rbf"]},
        }
    )
    assert space.names() == ["kernel", "scale", "threshold", "window"]
    assert space.defaults()["window"] == 20

    class FakeTrial:
        def __init__(self):
            self.calls = []

        def suggest_int(self, name, low, high, step=None):
            self.calls.append(name)
            return low

        def suggest_float(self, name, low, high, step=None, log=False):
            self.calls.append(name)
            return low

        def suggest_categorical(self, name, choices):
            self.calls.append(name)
            return choices[0]

    trial = FakeTrial()
    params = space.suggest(trial)  # type: ignore[arg-type]
    assert params["kernel"] == "linear"
    assert params["window"] == 5
    assert set(params) == set(space.names())


def test_invalid_space_raises():
    with pytest.raises(InvalidSearchSpaceError):
        make_search_space({"bad": {"kind": "weird"}})
    with pytest.raises(InvalidSearchSpaceError):
        make_search_space({"bad": {"kind": "int", "low": 10, "high": 1}})
    with pytest.raises(InvalidSearchSpaceError):
        make_search_space({"bad": {"kind": "loguniform", "low": 0.0, "high": 1.0}})
    with pytest.raises(InvalidSearchSpaceError):
        make_search_space({"bad": {"kind": "categorical", "choices": []}})


def test_empty_space_raises():
    with pytest.raises(InvalidSearchSpaceError):
        run_hpo(lambda params, trial: 1.0, SearchSpace({}), config=Config())


def test_run_hpo_maximizes():
    space = make_search_space({"x": {"kind": "float", "low": 0.0, "high": 1.0}})
    result = run_hpo(
        lambda params, trial: float(params["x"]),
        space,
        config=Config(seed=1, hpo_n_trials=6),
        direction="maximize",
    )
    assert result.n_completed == 6
    assert 0.0 <= result.best_value <= 1.0
    assert "x" in result.best_params
    assert result.to_dict()["direction"] == "maximize"


def test_run_hpo_error_policy_raise():
    space = make_search_space({"x": {"kind": "float", "low": 0.0, "high": 1.0}})

    def boom(params, trial):
        raise RuntimeError("always fails")

    with pytest.raises(StudyFailedError):
        run_hpo(boom, space, config=Config(seed=1, hpo_n_trials=2, strict=True))


def test_run_hpo_error_policy_skip():
    space = make_search_space({"x": {"kind": "float", "low": 0.0, "high": 1.0}})

    def sometimes(params, trial):
        if params["x"] < 0.2:
            raise RuntimeError("bad region")
        return params["x"]

    result = run_hpo(
        sometimes,
        space,
        config=Config(seed=1, hpo_n_trials=10, strict=False),
        on_error="skip",
    )
    assert result.best_value >= 0.2
    assert any(trial["state"] == "fail" for trial in result.trials)


def test_param_spec_direct():
    spec = ParamSpec(kind="int", low=1, high=10, default=5)
    space = SearchSpace({"a": spec})
    assert space.specs["a"] is spec
    assert space.to_dict()["a"]["default"] == 5
