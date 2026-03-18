"""Tests for zerokey.models.red_circle.RedCircle."""
from __future__ import annotations

from PIL import Image

from tests import make_pil_image


def test_estimate_radius_square() -> None:
    from zerokey.models.red_circle import RedCircle

    img = make_pil_image(400, 400)
    assert RedCircle.estimate_radius(img) == 10


def test_estimate_radius_rect() -> None:
    from zerokey.models.red_circle import RedCircle

    img = make_pil_image(200, 800)
    assert RedCircle.estimate_radius(img) == 5


def test_estimate_radius_small() -> None:
    from zerokey.models.red_circle import RedCircle

    img = make_pil_image(40, 40)
    assert RedCircle.estimate_radius(img) == 1


def _make_rc():
    from zerokey.models.red_circle import RedCircle

    return RedCircle.__new__(RedCircle)


def test_draw_circle_copy() -> None:
    rc = _make_rc()
    img = make_pil_image(100, 100)
    result = rc.draw_circle_on_image(img, 50, 50, radius=10, copy=True)
    assert result is not img


def test_draw_circle_no_copy() -> None:
    rc = _make_rc()
    img = make_pil_image(100, 100)
    result = rc.draw_circle_on_image(img, 50, 50, radius=10, copy=False)
    assert result is img


def test_draw_circle_modifies_pixels() -> None:
    rc = _make_rc()
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    original_data = list(img.getdata())  # type: ignore[arg-type]
    rc.draw_circle_on_image(img, 50, 50, radius=20, copy=False)
    assert list(img.getdata()) != original_data  # type: ignore[arg-type]
