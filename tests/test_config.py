"""配置与 ENV_CHANGEFORGE_* 覆盖测试。

作者: 晨星
"""

from __future__ import annotations

import json
import os
from dataclasses import fields

import pytest

from changeforge.core.config import ENV_PREFIX, Config, config_from_file, from_env, load_config
from changeforge.core.errors import InvalidConfigValue, UnknownConfigKey


def test_defaults():
    config = Config()
    assert config.seed == 42
    assert config.n_jobs == 1
    assert config.tolerance == 5
    assert config.output_dir == "artifacts"


def test_env_override_int_and_bool(tmp_path, monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}SEED", "7")
    monkeypatch.setenv(f"{ENV_PREFIX}TOLERANCE", "12")
    monkeypatch.setenv(f"{ENV_PREFIX}STRICT", "false")
    config = from_env()
    assert config.seed == 7
    assert config.tolerance == 12
    assert config.strict is False


def test_explicit_override_beats_env(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}SEED", "7")
    assert from_env(seed=99).seed == 99


def test_env_null_string_parses_to_none(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}HPO_TIMEOUT_S", "null")
    assert from_env().hpo_timeout_s is None
    monkeypatch.setenv(f"{ENV_PREFIX}HPO_TIMEOUT_S", "12.5")
    assert from_env().hpo_timeout_s == 12.5


def test_invalid_env_value_raises(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}SEED", "not-a-number")
    with pytest.raises(InvalidConfigValue) as excinfo:
        from_env()
    assert excinfo.value.code == "E101"


def test_unknown_key_raises():
    with pytest.raises(UnknownConfigKey):
        Config().with_overrides(no_such_key=1)


def test_none_override_is_ignored():
    assert Config().with_overrides(seed=None).seed == 42


def test_windows_n_jobs_clamped(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr("sys.platform", "win32")
    with pytest.warns(RuntimeWarning):
        config = Config(n_jobs=4)
    assert config.n_jobs == 1


def test_invalid_tolerance_raises():
    with pytest.raises(InvalidConfigValue):
        Config(tolerance=-1)


def test_config_from_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"seed": 123, "tolerance": 3}), encoding="utf-8")
    config = config_from_file(str(path))
    assert config.seed == 123
    assert config.tolerance == 3


def test_config_from_bad_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(InvalidConfigValue):
        config_from_file(str(path))


def test_to_dict_roundtrip():
    payload = load_config(seed=5).to_dict()
    assert payload["seed"] == 5
    assert set(payload) == {field.name for field in fields(Config)}
