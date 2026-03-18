"""Tests for patchalign3d/tools/eval_cli.py — text cleaning, projector, and model builder."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tools.eval_cli import PatchToTextProj, _clean_text, build_model  # type: ignore[import-untyped]

from tests import assert_tensor_close


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

def test_clean_text_lowercase() -> None:
    assert _clean_text("Hello World") == "hello world"


def test_clean_text_underscores() -> None:
    assert _clean_text("hello_world") == "hello world"


def test_clean_text_punctuation() -> None:
    assert _clean_text("hello, world!") == "hello world"


def test_clean_text_whitespace() -> None:
    assert _clean_text("  hello   world  ") == "hello world"


def test_clean_text_combined() -> None:
    assert _clean_text("Hello_World! Foo") == "hello world foo"


def test_clean_text_empty() -> None:
    assert _clean_text("") == ""


# ---------------------------------------------------------------------------
# PatchToTextProj
# ---------------------------------------------------------------------------

def test_proj_output_shape() -> None:
    proj = PatchToTextProj(384, 512)
    x = torch.randn(2, 384, 128)
    out = proj(x)
    assert out.shape == (2, 128, 512)


def test_proj_l2_normalized() -> None:
    proj = PatchToTextProj(384, 512)
    x = torch.randn(2, 384, 128)
    out = proj(x)
    norms = out.norm(dim=-1)
    assert_tensor_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_proj_linear_params() -> None:
    proj = PatchToTextProj(384, 512)
    assert isinstance(proj.proj, nn.Linear)
    assert proj.proj.in_features == 384
    assert proj.proj.out_features == 512


def test_proj_forward_backward() -> None:
    proj = PatchToTextProj(64, 32)
    x = torch.randn(1, 64, 16, requires_grad=True)
    out = proj(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

_BUILD_MODEL_AVAILABLE = True
try:
    _test_model = build_model(arch="pointtransformer")
except Exception:
    _BUILD_MODEL_AVAILABLE = False


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_model(arch="nonexistent")


@pytest.mark.skipif(not _BUILD_MODEL_AVAILABLE, reason="PointTransformer model imports unavailable")
def test_build_model_known_arch() -> None:
    model = build_model(arch="pointtransformer")
    assert isinstance(model, nn.Module)
