"""错误码体系测试。

作者: 晨星
"""

from __future__ import annotations

import pytest

from changeforge.core.errors import (
    ERROR_CLASSES,
    ChangeForgeError,
    InvalidConfigValue,
    InvalidSignalError,
    get_error_class,
)


def test_error_codes_cover_e100_to_e500():
    codes = {int(code[1:]) for code in ERROR_CLASSES}
    assert 100 in codes and 500 in codes
    # E000 是基类保留码，不参与业务分段校验
    # E5xx 段（流水线/评测/基准/CLI）目前最大到 E503，故上界放宽到 599
    assert all(100 <= code <= 599 for code in codes if code != 0)
    assert sorted(codes - {0})[0] == 100


def test_render_includes_code_and_context():
    error = InvalidConfigValue("种子非法", key="seed", value=-1)
    text = error.render()
    assert "[E101]" in text
    assert "key='seed'" in text
    assert isinstance(error, ChangeForgeError)


def test_default_message_used_when_message_empty():
    error = InvalidSignalError()
    assert "E201" in error.render()
    assert error.message == error.default_message


def test_to_dict_shape():
    payload = InvalidConfigValue("bad", key="seed").to_dict()
    assert payload["code"] == "E101"
    assert payload["context"] == {"key": "seed"}


def test_get_error_class_unknown_falls_back():
    assert get_error_class("E999") is ChangeForgeError
    assert get_error_class("e201") is InvalidSignalError


def test_raise_and_catch_as_base():
    with pytest.raises(ChangeForgeError) as excinfo:
        raise InvalidSignalError("空信号", name="s")
    assert excinfo.value.code == "E201"
