"""Tests for zerokey.models.gpt4o.GPT4o."""
from __future__ import annotations

import base64
import io
from unittest import mock

from PIL import Image

from tests import make_pil_image


def test_pil_to_base64_returns_string() -> None:
    from zerokey.models.gpt4o import GPT4o

    img = make_pil_image(32, 32)
    result = GPT4o.pil_to_base64(img)
    assert isinstance(result, str)


def test_pil_to_base64_valid_base64() -> None:
    from zerokey.models.gpt4o import GPT4o

    img = make_pil_image(32, 32)
    result = GPT4o.pil_to_base64(img)
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_pil_to_base64_roundtrip() -> None:
    from zerokey.models.gpt4o import GPT4o

    img = make_pil_image(32, 32)
    result = GPT4o.pil_to_base64(img)
    decoded = base64.b64decode(result)
    loaded = Image.open(io.BytesIO(decoded))
    assert loaded.format == "PNG"
    assert loaded.size == (32, 32)


def test_pil_to_base64_png_header() -> None:
    from zerokey.models.gpt4o import GPT4o

    img = make_pil_image(32, 32)
    result = GPT4o.pil_to_base64(img)
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def _make_gpt():
    from zerokey.models.gpt4o import GPT4o

    return GPT4o.__new__(GPT4o)


def test_iter_over_list_flat_dict() -> None:
    gpt = _make_gpt()
    result = list(gpt.iter_over_list({"a": "x", "b": "y"}))
    assert result == ["x", "y"]


def test_iter_over_list_nested_dict() -> None:
    gpt = _make_gpt()
    result = list(gpt.iter_over_list({"a": {"b": "x"}}))
    assert result == ["x"]


def test_iter_over_list_with_list() -> None:
    gpt = _make_gpt()
    result = list(gpt.iter_over_list({"a": ["x", "y"]}))
    assert result == ["x", "y"]


def test_iter_over_list_with_set() -> None:
    gpt = _make_gpt()
    result = set(gpt.iter_over_list({"a": {1, 2}}))
    assert result == {1, 2}


def test_iter_over_list_deep_nesting() -> None:
    gpt = _make_gpt()
    data = {"a": {"b": {"c": "deep"}}}
    result = list(gpt.iter_over_list(data))
    assert result == ["deep"]


def test_iter_over_list_mixed() -> None:
    gpt = _make_gpt()
    data = {"a": ["x", "y"], "b": "z", "c": {"d": "w"}}
    result = list(gpt.iter_over_list(data))
    assert result == ["x", "y", "z", "w"]


@mock.patch("zerokey.models.gpt4o.OpenAI")
def test_init_creates_client(mock_openai_cls: mock.MagicMock) -> None:
    from zerokey.models.gpt4o import GPT4o

    gpt = GPT4o()
    mock_openai_cls.assert_called_once()
    assert gpt.client is mock_openai_cls.return_value


@mock.patch("zerokey.models.gpt4o.OpenAI")
def test_get_kplist_calls_chat_api(mock_openai_cls: mock.MagicMock) -> None:
    from zerokey.models.gpt4o import GPT4o

    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create.return_value = mock.MagicMock()

    gpt = GPT4o()
    img = make_pil_image(32, 32)
    gpt.get_kplist(img)

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("model", call_kwargs[1].get("model")) == "gpt-4o"
