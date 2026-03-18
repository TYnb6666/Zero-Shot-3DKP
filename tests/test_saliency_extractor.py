"""Tests for feature_backprojection/saliency_extractor.py — str2bool helper."""
from __future__ import annotations

import argparse

import pytest

from feature_backprojection.saliency_extractor import str2bool


@pytest.mark.parametrize("v", ["yes", "true", "t", "y", "1"])
def test_true_values(v: str) -> None:
    assert str2bool(v)


@pytest.mark.parametrize("v", ["no", "false", "f", "n", "0"])
def test_false_values(v: str) -> None:
    assert not str2bool(v)


def test_case_insensitive() -> None:
    assert str2bool("YES")
    assert str2bool("True")
    assert not str2bool("FALSE")


def test_bool_passthrough_true() -> None:
    assert str2bool(True)


def test_bool_passthrough_false() -> None:
    assert not str2bool(False)


def test_invalid_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


def test_empty_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("")


def test_mixed_case() -> None:
    assert str2bool("TrUe")
