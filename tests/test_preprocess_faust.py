"""Tests for patchalign3d/tools/preprocess_faust_partnete.py — FPS sampling."""
from __future__ import annotations

import numpy as np
import pytest

from tools.preprocess_faust_partnete import fps_indices  # type: ignore[import-untyped]


@pytest.fixture
def pts() -> np.ndarray:
    np.random.seed(42)
    return np.random.randn(200, 3).astype(np.float64)


def test_correct_count(pts: np.ndarray) -> None:
    idx = fps_indices(pts, 50)
    assert len(idx) == 50


def test_valid_indices(pts: np.ndarray) -> None:
    idx = fps_indices(pts, 50)
    assert np.all(idx >= 0)
    assert np.all(idx < 200)


def test_no_duplicates(pts: np.ndarray) -> None:
    idx = fps_indices(pts, 50)
    assert len(np.unique(idx)) == 50


def test_k_equals_n(pts: np.ndarray) -> None:
    idx = fps_indices(pts, 200)
    np.testing.assert_array_equal(idx, np.arange(200, dtype=np.int64))


def test_k_greater_than_n(pts: np.ndarray) -> None:
    idx = fps_indices(pts, 300)
    np.testing.assert_array_equal(idx, np.arange(200, dtype=np.int64))


def test_deterministic(pts: np.ndarray) -> None:
    idx1 = fps_indices(pts, 50, seed=7)
    idx2 = fps_indices(pts, 50, seed=7)
    np.testing.assert_array_equal(idx1, idx2)


def test_different_seeds(pts: np.ndarray) -> None:
    idx1 = fps_indices(pts, 50, seed=0)
    idx2 = fps_indices(pts, 50, seed=99)
    assert not np.array_equal(idx1, idx2)


def test_spatial_coverage(pts: np.ndarray) -> None:
    """Selected points should span most of the bounding box."""
    idx = fps_indices(pts, 50, seed=0)
    selected = pts[idx]
    all_min = pts.min(axis=0)
    all_max = pts.max(axis=0)
    sel_min = selected.min(axis=0)
    sel_max = selected.max(axis=0)
    span = all_max - all_min
    for d in range(3):
        assert sel_min[d] <= all_min[d] + 0.3 * span[d]
        assert sel_max[d] >= all_max[d] - 0.3 * span[d]
