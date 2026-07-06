"""Utilities for saved PointNeXt surface-anchor feature artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

REQUIRED_KEYS = {
    "schema_version",
    "mesh_id",
    "category",
    "surface_xyz",
    "surface_feat",
    "surface_normal",
    "sampling_info",
    "model_info",
    "norm_info",
}


def load_surface_features(pt_path: str | Path) -> dict[str, Any]:
    """Load a PointNeXt surface feature artifact on CPU and validate it."""
    path = Path(pt_path)
    data = torch.load(path, map_location="cpu")
    validate_surface_features(data, source=path)
    return data


def validate_surface_features(data: Any, source: str | Path = "<memory>") -> None:
    """Validate the minimal contract required for candidate snapping.

    The saved artifact must contain the exact fixed 4096 anchors used for
    feature extraction.  Downstream code is expected to snap candidates only to
    ``surface_xyz`` and inherit features from the same row in ``surface_feat``.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected dict artifact, got {type(data).__name__}")
    missing = REQUIRED_KEYS.difference(data)
    if missing:
        raise ValueError(f"{source}: missing required keys {sorted(missing)}")

    surface_xyz = data["surface_xyz"]
    surface_feat = data["surface_feat"]
    surface_normal = data["surface_normal"]
    for key, tensor in (
        ("surface_xyz", surface_xyz),
        ("surface_feat", surface_feat),
        ("surface_normal", surface_normal),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{source}: {key} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{source}: {key} must be float32, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{source}: {key} contains non-finite values")

    if tuple(surface_xyz.shape) != (4096, 3):
        raise ValueError(f"{source}: surface_xyz must have shape [4096, 3], got {tuple(surface_xyz.shape)}")
    if surface_feat.ndim != 2 or surface_feat.shape[0] != 4096:
        raise ValueError(f"{source}: surface_feat must have shape [4096, D], got {tuple(surface_feat.shape)}")
    if tuple(surface_normal.shape) != (4096, 3):
        raise ValueError(f"{source}: surface_normal must have shape [4096, 3], got {tuple(surface_normal.shape)}")

    model_info = data.get("model_info", {})
    feature_dim = model_info.get("feature_dim") if isinstance(model_info, dict) else None
    if feature_dim is not None and int(feature_dim) != int(surface_feat.shape[1]):
        raise ValueError(
            f"{source}: model_info.feature_dim={feature_dim} does not match surface_feat dim {surface_feat.shape[1]}"
        )


def surface_feature_path(root: str | Path, category: str, mesh_id: str) -> Path:
    """Return the canonical per-mesh PointNeXt surface feature path."""
    return Path(root) / category / f"{mesh_id}.pt"
