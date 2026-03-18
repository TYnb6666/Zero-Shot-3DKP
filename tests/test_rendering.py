"""Tests for the debug_enabled() helper in zerokey/rendering.py."""
from __future__ import annotations

import os
from unittest.mock import patch

from zerokey.rendering import debug_enabled


def test_debug_disabled_by_default() -> None:
    with patch.dict(os.environ, {}, clear=True), \
         patch("sys.gettrace", return_value=None):
        assert not debug_enabled()


def test_debug_enabled_env_1() -> None:
    with patch.dict(os.environ, {"DEBUG": "1"}):
        assert debug_enabled()


def test_debug_enabled_env_true() -> None:
    with patch.dict(os.environ, {"DEBUG": "true"}):
        assert debug_enabled()


def test_debug_enabled_env_yes() -> None:
    with patch.dict(os.environ, {"DEBUG": "yes"}):
        assert debug_enabled()


def test_debug_enabled_gettrace() -> None:
    with patch.dict(os.environ, {}, clear=True), \
         patch("sys.gettrace", return_value=lambda: None):
        assert debug_enabled()


def test_debug_disabled_env_empty() -> None:
    with patch.dict(os.environ, {"DEBUG": ""}), \
         patch("sys.gettrace", return_value=None):
        assert not debug_enabled()
