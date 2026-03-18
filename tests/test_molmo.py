"""Tests for zerokey.models.molmo.Molmo."""
from __future__ import annotations

from unittest import mock
from xml.etree.ElementTree import ParseError

import pytest
from PIL import Image

from tests import make_pil_image


def test_color_names_is_tuple() -> None:
    from zerokey.models.molmo import Molmo

    assert isinstance(Molmo.COLOR_NAMES, tuple)


def test_color_names_length_10() -> None:
    from zerokey.models.molmo import Molmo

    assert len(Molmo.COLOR_NAMES) == 10


def test_color_values_length_10() -> None:
    from zerokey.models.molmo import Molmo

    assert len(Molmo.COLOR_VALUES) == 10


def test_color_values_not_none() -> None:
    from zerokey.models.molmo import Molmo

    for i, v in enumerate(Molmo.COLOR_VALUES):
        assert v is not None, f"COLOR_VALUES[{i}] is None"


def test_parse_points_single() -> None:
    from zerokey.models.molmo import Molmo

    kps, alt = Molmo.parse_points_str('<points x1="23.4" y1="56.7" alt="nose"/>')
    assert alt == "nose"
    assert "1" in kps
    assert abs(kps["1"]["x"] - 23.4) < 1e-5
    assert abs(kps["1"]["y"] - 56.7) < 1e-5


def test_parse_points_multiple() -> None:
    from zerokey.models.molmo import Molmo

    kps, _alt = Molmo.parse_points_str('<points x1="10" y1="20" x2="30" y2="40"/>')
    assert len(kps) == 2
    assert abs(kps["1"]["x"] - 10.0) < 1e-5
    assert abs(kps["2"]["y"] - 40.0) < 1e-5


def test_parse_points_no_alt() -> None:
    from zerokey.models.molmo import Molmo

    _kps, alt = Molmo.parse_points_str('<points x1="10" y1="20"/>')
    assert alt is None


def test_parse_points_invalid_xml() -> None:
    from zerokey.models.molmo import Molmo

    with pytest.raises(ParseError):
        Molmo.parse_points_str("not xml")


def test_parse_points_empty_element() -> None:
    from zerokey.models.molmo import Molmo

    kps, alt = Molmo.parse_points_str("<points/>")
    assert len(kps) == 0
    assert alt is None


def test_draw_points_returns_image() -> None:
    from zerokey.models.molmo import Molmo

    img = make_pil_image(100, 100)
    kps = {"1": {"x": 50.0, "y": 50.0}}
    result = Molmo.draw_points(img, kps)
    assert isinstance(result, Image.Image)


def test_draw_points_modifies_image() -> None:
    from zerokey.models.molmo import Molmo

    img = make_pil_image(100, 100)
    original = img.copy()
    kps = {"1": {"x": 50.0, "y": 50.0}}
    result = Molmo.draw_points(img, kps)
    assert result is img
    assert list(img.getdata()) != list(original.getdata())  # type: ignore[arg-type]


def test_draw_points_scaling() -> None:
    from zerokey.models.molmo import Molmo

    img = make_pil_image(100, 100)
    original = img.copy()
    kps = {"1": {"x": 100.0, "y": 100.0}}
    Molmo.draw_points(img, kps, kps_wh=(200, 200), radius=5, width=2)
    assert list(img.getdata()) != list(original.getdata())  # type: ignore[arg-type]


def test_draw_points_alpha_mode() -> None:
    from zerokey.models.molmo import Molmo

    img = make_pil_image(100, 100)
    kps = {"1": {"x": 50.0, "y": 50.0}}
    result = Molmo.draw_points(img, kps, radius=10, width=None)
    assert "A" in result.mode


def test_draw_points_no_radius() -> None:
    from zerokey.models.molmo import Molmo

    img = make_pil_image(100, 100)
    original = img.copy()
    kps = {"1": {"x": 50.0, "y": 50.0}}
    Molmo.draw_points(img, kps, radius=0)
    assert list(img.getdata()) != list(original.getdata())  # type: ignore[arg-type]


@mock.patch("transformers.AutoConfig")
@mock.patch("zerokey.models.molmo.AutoProcessor")
@mock.patch("zerokey.models.molmo.AutoModelForCausalLM")
def test_init_calls_from_pretrained(
    mock_model_cls: mock.MagicMock,
    mock_proc_cls: mock.MagicMock,
    mock_config_cls: mock.MagicMock,
) -> None:
    from zerokey.models.molmo import Molmo

    Molmo("test/model-path")
    mock_model_cls.from_pretrained.assert_called_once()
    mock_proc_cls.from_pretrained.assert_called_once()
    assert mock_model_cls.from_pretrained.call_args[0][0] == "test/model-path"


@mock.patch("transformers.AutoConfig")
@mock.patch("zerokey.models.molmo.AutoProcessor")
@mock.patch("zerokey.models.molmo.AutoModelForCausalLM")
def test_generated_kps_points_returns_string(
    mock_model_cls: mock.MagicMock,
    mock_proc_cls: mock.MagicMock,
    mock_config_cls: mock.MagicMock,
) -> None:
    from zerokey.models.molmo import Molmo

    mock_processor = mock.MagicMock()
    mock_proc_cls.from_pretrained.return_value = mock_processor
    mock_processor.process.return_value = {
        "input_ids": mock.MagicMock(**{"size.return_value": 5}),  # type: ignore[call-overload]
        "images": mock.MagicMock(),
    }
    for v in mock_processor.process.return_value.values():
        v.to.return_value = v
        v.unsqueeze.return_value = v

    mock_model = mock.MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_output = mock.MagicMock()
    mock_output.sequences = mock.MagicMock()
    mock_output.sequences.__getitem__ = mock.MagicMock(return_value=mock.MagicMock())
    mock_model.generate_from_batch.return_value = mock_output
    mock_processor.tokenizer.decode.return_value = "<points x1='10' y1='20'/>"

    molmo = Molmo("test/path")
    img = make_pil_image(64, 64)
    result = molmo.generated_kps_points(img, "find keypoints")
    assert isinstance(result, str)


@mock.patch("transformers.AutoConfig")
@mock.patch("zerokey.models.molmo.AutoProcessor")
@mock.patch("zerokey.models.molmo.AutoModelForCausalLM")
def test_image_text_token_calls_processor(
    mock_model_cls: mock.MagicMock,
    mock_proc_cls: mock.MagicMock,
    mock_config_cls: mock.MagicMock,
) -> None:
    from zerokey.models.molmo import Molmo

    mock_processor = mock.MagicMock()
    mock_proc_cls.from_pretrained.return_value = mock_processor
    mock_processor.process.return_value = {"input_ids": mock.MagicMock()}

    molmo = Molmo("test/path")
    img = make_pil_image(64, 64)
    molmo.image_text_token(img, "test prompt")
    mock_processor.process.assert_called_once()
