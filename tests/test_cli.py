"""CLI 测试。

作者: 晨星
"""

from __future__ import annotations

import json

import pytest

from changeforge.cli import main


def test_info(capsys):
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["author"] == "晨星"
    assert "config" in payload


def test_list_datasets(capsys):
    assert main(["list-datasets"]) == 0
    assert "constant" in capsys.readouterr().out


def test_list_detectors_when_empty(empty_registry, capsys):
    assert main(["list-detectors"]) == 0
    assert "注册表为空" in capsys.readouterr().out


def test_list_detectors_with_dummy(dummy_detector_module, capsys):
    assert main(["list-detectors"]) == 0
    assert "dummy" in capsys.readouterr().out


def test_run_requires_detectors(empty_registry, capsys):
    assert main(["run", "--n-samples", "200"]) == 1
    assert "E503" in capsys.readouterr().err


def test_run_with_dummy(dummy_detector_module, capsys):
    assert main(["run", "--n-samples", "300", "--detector", "dummy"]) == 0
    output = capsys.readouterr().out
    assert "predicted" in output


def test_run_json(dummy_detector_module, capsys):
    assert main(["run", "--n-samples", "300", "--detector", "dummy", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["detector_name"] == "dummy"


def test_gen_writes_csv(tmp_path, capsys):
    out = str(tmp_path / "gen.csv")
    assert main(["gen", "--n-samples", "120", "--n-change-points", "2", "--out", out]) == 0
    assert "已生成" in capsys.readouterr().out
    with open(out, encoding="utf-8") as handle:
        header = handle.readline().strip()
    assert header.endswith("is_change")


def test_benchmark_writes_json(tmp_path, dummy_detector_module, capsys):
    out = str(tmp_path / "bench.json")
    code = main(
        [
            "benchmark",
            "--n-samples",
            "300",
            "--n-change-points",
            "2",
            "--kinds",
            "constant,variance",
            "--detectors",
            "dummy",
            "--out",
            out,
        ]
    )
    assert code == 0
    with open(out, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["n_records"] == 2
    assert "benchmark.json" in capsys.readouterr().out


def test_env_override_via_cli(monkeypatch, capsys):
    monkeypatch.setenv("ENV_CHANGEFORGE_TOLERANCE", "11")
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["tolerance"] == 11


def test_bad_config_file(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{oops", encoding="utf-8")
    assert main(["--config", str(path), "info"]) == 2


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["no-such-command"])
