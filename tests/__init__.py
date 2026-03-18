"""Shared helpers for ZeroKey tests."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator
from unittest import mock

import numpy as np
import torch
from PIL import Image

HAS_CUDA = torch.cuda.is_available()


def make_pil_image(w: int = 64, h: int = 64, mode: str = "RGB") -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def make_random_points(n: int = 100, dim: int = 3) -> "np.ndarray[Any, Any]":
    return np.random.randn(n, dim).astype(np.float32)


def assert_tensor_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Tensors not close:\n  a={a}\n  b={b}"
    )


@contextmanager
def env_override(**kwargs: str) -> Generator[None, None, None]:
    """Temporarily override environment variables."""
    import os
    with mock.patch.dict(os.environ, kwargs):
        yield
