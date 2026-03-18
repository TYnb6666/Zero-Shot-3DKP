"""Tests for zerokey/_defaults.py — env-var-backed path constants."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import zerokey._defaults as _defaults_mod


@pytest.fixture(autouse=True)
def _restore_defaults():
    yield
    importlib.reload(_defaults_mod)


def test_default_log_dir_is_path() -> None:
    assert isinstance(_defaults_mod.DEFAULT_LOG_DIR, Path)


def test_default_log_dir_value() -> None:
    assert str(_defaults_mod.DEFAULT_LOG_DIR).endswith("zerokey-results"), (
        f"Expected path ending with 'zerokey-results', got {_defaults_mod.DEFAULT_LOG_DIR}"
    )


def test_keypoint_dataset_path_default() -> None:
    assert _defaults_mod.KEYPOINT_DATASET_PATH == "keypointnet"


def test_colmap_data_path_default() -> None:
    assert _defaults_mod.COLMAP_DATA_PATH == ""


def test_pointbert_ckpt_default() -> None:
    assert _defaults_mod.POINTBERT_CKPT == "model_ckpt/2stagemodel.pt"


def test_env_override_log_dir() -> None:
    with patch.dict(os.environ, {"ZEROKEY_LOG_DIR": "/tmp/test-log"}):
        importlib.reload(_defaults_mod)
        assert _defaults_mod.DEFAULT_LOG_DIR == Path("/tmp/test-log")


def test_env_override_keypoint_path() -> None:
    with patch.dict(os.environ, {"KEYPOINT_DATASET_PATH": "/tmp/kp"}):
        importlib.reload(_defaults_mod)
        assert _defaults_mod.KEYPOINT_DATASET_PATH == "/tmp/kp"


def test_all_constants_are_str_or_path() -> None:
    """Every public, uppercase constant should be either str or Path."""
    for name in dir(_defaults_mod):
        if name.startswith("_") or not name.isupper():
            continue
        value = getattr(_defaults_mod, name)
        assert isinstance(value, (str, Path)), (
            f"{name} has unexpected type {type(value).__name__}"
        )
