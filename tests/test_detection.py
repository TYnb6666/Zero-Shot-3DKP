"""Tests for COLOR_NAMES and COLOR_MAP on KeypointDetectionMixin in zerokey/_detection.py."""
from __future__ import annotations

from collections import OrderedDict

import pytest
from PIL import ImageColor

from zerokey._detection import KeypointDetectionMixin


def test_color_names_is_ordered_dict() -> None:
    assert isinstance(KeypointDetectionMixin.COLOR_NAMES, OrderedDict)


def test_color_names_nonempty() -> None:
    assert len(KeypointDetectionMixin.COLOR_NAMES) >= 100


def test_color_names_values_are_valid_hex() -> None:
    for name, hex_val in KeypointDetectionMixin.COLOR_NAMES.items():
        try:
            rgb = ImageColor.getrgb(hex_val)
        except ValueError:
            pytest.fail(f"COLOR_NAMES[{name!r}] = {hex_val!r} is not a valid color string")
        assert len(rgb) == 3, f"Expected 3-tuple for {name}, got {rgb}"


def test_color_names_no_black() -> None:
    for name, hex_val in KeypointDetectionMixin.COLOR_NAMES.items():
        rgb = ImageColor.getrgb(hex_val)
        assert rgb != (0, 0, 0), (
            f"COLOR_NAMES should not contain pure black but found {name!r}"
        )


def test_color_map_keys_are_3_bytes() -> None:
    for key in KeypointDetectionMixin.COLOR_MAP:
        assert isinstance(key, bytes)
        assert len(key) == 3, f"Key {key!r} is not 3 bytes"


def test_color_map_values_sequential() -> None:
    """Values should be a subset of 0..len(COLOR_NAMES)-1, strictly increasing when sorted."""
    values = sorted(KeypointDetectionMixin.COLOR_MAP.values())
    assert min(values) >= 0
    assert max(values) < len(KeypointDetectionMixin.COLOR_NAMES)
    assert len(values) == len(set(values))


def test_color_map_same_length_as_names() -> None:
    """COLOR_MAP may be smaller than COLOR_NAMES due to hex collisions (aliases),
    but should not exceed it."""
    assert len(KeypointDetectionMixin.COLOR_MAP) <= len(KeypointDetectionMixin.COLOR_NAMES)


def test_color_map_roundtrip() -> None:
    """Every hex value in COLOR_NAMES should produce a key present in COLOR_MAP."""
    for name, hex_val in KeypointDetectionMixin.COLOR_NAMES.items():
        key = bytes.fromhex(str(hex_val).lstrip("#"))
        assert key in KeypointDetectionMixin.COLOR_MAP, (
            f"Hex {hex_val!r} (name={name!r}) not found in COLOR_MAP"
        )
