"""Tests for patchalign3d/tools/dump_matching_patch_features.py — checkpoint dim inference."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from tools.dump_matching_patch_features import infer_proj_dims  # type: ignore[import-untyped]


def test_valid_checkpoint() -> None:
    """A well-formed checkpoint should yield dimensions from the weight tensor."""
    weight = torch.randn(256, 384)
    ckpt = {"proj": {"proj.weight": weight}}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        try:
            in_dim, out_dim = infer_proj_dims(Path(f.name))
            assert in_dim == 384
            assert out_dim == 256
        finally:
            os.unlink(f.name)


def test_missing_proj_key() -> None:
    """Checkpoint without a 'proj' key should return defaults."""
    ckpt = {"model": {"layer.weight": torch.randn(10, 10)}}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        try:
            assert infer_proj_dims(Path(f.name)) == (384, 512)
        finally:
            os.unlink(f.name)


def test_nonexistent_file() -> None:
    """A path that does not exist should return defaults."""
    result = infer_proj_dims(Path("/tmp/_nonexistent_checkpoint_abc123.pt"))
    assert result == (384, 512)


def test_custom_defaults() -> None:
    """Custom default_in / default_out should be returned on failure."""
    result = infer_proj_dims(
        Path("/tmp/_nonexistent_checkpoint_abc123.pt"),
        default_in=100,
        default_out=200,
    )
    assert result == (100, 200)


def test_corrupt_file() -> None:
    """A file with random bytes should return defaults."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(os.urandom(256))
        f.flush()
        try:
            assert infer_proj_dims(Path(f.name)) == (384, 512)
        finally:
            os.unlink(f.name)
