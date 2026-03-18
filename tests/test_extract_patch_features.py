"""Tests for patchalign3d/inference/extract_patch_features.py — I/O helpers and resampling."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from inference.extract_patch_features import (  # type: ignore[import-untyped]
    _load_npy,
    _load_npz,
    _load_txt,
    load_point_cloud,
    resample_points,
)


def test_load_npz_with_points_key() -> None:
    pts = np.random.randn(50, 3).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, points=pts)
        try:
            loaded = _load_npz(Path(f.name))
            np.testing.assert_array_equal(loaded, pts)
        finally:
            os.unlink(f.name)


def test_load_npz_with_xyz_key() -> None:
    pts = np.random.randn(40, 4).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, xyz=pts)
        try:
            loaded = _load_npz(Path(f.name))
            np.testing.assert_array_equal(loaded, pts)
        finally:
            os.unlink(f.name)


def test_load_npz_single_array() -> None:
    pts = np.random.randn(30, 3).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, my_array=pts)
        try:
            loaded = _load_npz(Path(f.name))
            np.testing.assert_array_equal(loaded, pts)
        finally:
            os.unlink(f.name)


def test_load_npy() -> None:
    pts = np.random.randn(60, 3).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, pts)
        try:
            loaded = _load_npy(Path(f.name))
            np.testing.assert_array_equal(loaded, pts)
        finally:
            os.unlink(f.name)


def test_load_txt() -> None:
    pts = np.random.randn(20, 3)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        np.savetxt(f.name, pts)
        try:
            loaded = _load_txt(Path(f.name))
            np.testing.assert_allclose(loaded, pts, atol=1e-5)
        finally:
            os.unlink(f.name)


def test_load_point_cloud_npz() -> None:
    pts = np.random.randn(50, 3).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, points=pts)
        try:
            loaded = load_point_cloud(f.name)
            assert loaded.shape == (50, 3)
            assert loaded.dtype == np.float32
        finally:
            os.unlink(f.name)


def test_load_point_cloud_npy() -> None:
    pts = np.random.randn(40, 4).astype(np.float64)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, pts)
        try:
            loaded = load_point_cloud(f.name)
            assert loaded.shape == (40, 4)
            assert loaded.dtype == np.float32
        finally:
            os.unlink(f.name)


def test_load_point_cloud_unsupported() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b"{}")
        f.flush()
        try:
            with pytest.raises(ValueError):
                load_point_cloud(f.name)
        finally:
            os.unlink(f.name)


def test_load_point_cloud_nonexistent() -> None:
    with pytest.raises(FileNotFoundError):
        load_point_cloud("/tmp/_nonexistent_pc_abc123.npz")


def test_load_point_cloud_ensures_float32() -> None:
    pts = np.random.randn(30, 3).astype(np.float64)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, pts)
        try:
            loaded = load_point_cloud(f.name)
            assert loaded.dtype == np.float32
        finally:
            os.unlink(f.name)


def test_load_point_cloud_min_3_columns() -> None:
    pts = np.random.randn(30, 2).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, pts)
        try:
            with pytest.raises(ValueError):
                load_point_cloud(f.name)
        finally:
            os.unlink(f.name)


def test_resample_exact() -> None:
    pts = np.random.randn(100, 3).astype(np.float32)
    out, idx = resample_points(pts, 100)
    assert out.shape == (100, 3)
    np.testing.assert_array_equal(idx, np.arange(100, dtype=np.int64))


def test_resample_downsample() -> None:
    pts = np.random.randn(200, 3).astype(np.float32)
    out, idx = resample_points(pts, 50)
    assert out.shape == (50, 3)
    assert len(idx) == 50
    assert np.all(idx >= 0)
    assert np.all(idx < 200)


def test_resample_upsample() -> None:
    pts = np.random.randn(30, 3).astype(np.float32)
    out, idx = resample_points(pts, 100)
    assert out.shape == (100, 3)
    assert len(idx) == 100
    # All original indices should appear at least once.
    assert set(range(30)).issubset(set(idx.tolist()))
