"""Tests for patchalign3d/tools/seen_unseen_objaverse_general.py — slug parsing."""
from __future__ import annotations

from tools.seen_unseen_objaverse_general import _category_from_slug  # type: ignore[import-untyped]


def test_office_chair_12() -> None:
    assert _category_from_slug("office_chair_12") == "office_chair"


def test_mug() -> None:
    assert _category_from_slug("mug") == "mug"


def test_multi_underscore() -> None:
    assert _category_from_slug("big_red_chair_99") == "big_red_chair"


def test_trailing_number() -> None:
    assert _category_from_slug("table_0") == "table"


def test_no_number_with_underscore() -> None:
    assert _category_from_slug("hello_world") == "hello"
