"""Checkpoint inspection utilities for the patch feature pipeline.

Provides :func:`infer_proj_dims` which peeks inside a saved checkpoint to
determine the input/output dimensions of the ``PatchToTextProj`` linear
layer, so the caller can construct a matching module before loading
weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def infer_proj_dims(
    ckpt_path: Path,
    default_in: int = 384,
    default_out: int = 512,
) -> Tuple[int, int]:
    """Inspect a checkpoint to infer the projector's ``(in_dim, out_dim)``.

    Looks for a ``proj.weight`` tensor inside the ``'proj'`` sub-dict of
    the checkpoint.  If found, the weight shape ``(out_dim, in_dim)`` is
    returned; otherwise the supplied defaults are used.

    Args:
        ckpt_path: Path to a ``.pt`` checkpoint file.
        default_in: Fallback input dimension.
        default_out: Fallback output dimension.

    Returns:
        ``(in_dim, out_dim)`` tuple.
    """
    try:
        ck = torch.load(str(ckpt_path), map_location='cpu')
        if isinstance(ck, dict) and 'proj' in ck and isinstance(ck['proj'], dict):
            w = ck['proj'].get('proj.weight', None)
            if w is not None and hasattr(w, 'shape'):
                return int(w.shape[1]), int(w.shape[0])
    except Exception:
        pass
    return int(default_in), int(default_out)
