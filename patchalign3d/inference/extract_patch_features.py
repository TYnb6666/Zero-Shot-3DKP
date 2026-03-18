#!/usr/bin/env python3
"""
Extract patch features from a point cloud using a trained 2-stage model.

Inputs:
  - A point cloud file (.npz/.npy/.txt/.xyz; optional .ply/.glb via trimesh)
  - A checkpoint that contains at least { 'model': state_dict }. If it also
    contains 'proj', projected features will be produced as well.

Outputs:
  - An .npz (default) or .pt file containing:
      patch_emb:    (G, C)   raw patch embeddings (C≈384)
      patch_feat:   (G, D)   projected + L2-normalized features (if 'proj' found)
      patch_centers:(G, 3)   patch centroid xyz
      patch_indices:(G, M)   membership indices into the resampled point cloud
      points_sampled:(N, C0) resampled input points actually fed to the model
      sample_to_input_idx:(N,) mapping from sampled points to original input indices

Usage example:
  python -m segmentation.inference.extract_patch_features \
    --input /path/to/pointcloud.npz \
    --ckpt  $POINTBERT_CKPT \
    --output /tmp/pointcloud_patches.npz

Notes:
  - Matches evaluation/training preprocessing: swap Y/Z axes then pass (B,C,N).
  - Defaults to the PointTransformer-based encoder with num_group=128, group_size=32.
  - Uses Farthest Point Sampling (FPS) for point cloud downsampling (reuses existing utilities).
  - Requires CUDA for KNN grouping (knn_cuda). If unavailable, this script will
    error out when calling forward_patches. Use a CUDA environment with knn_cuda installed.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import fire
import numpy as np
import torch

# Make segmentation repo modules importable (models, etc.)

# Get segmentation directory path
_SEG_DIR = Path(__file__).resolve().parent.parent

# Import reusable utilities.
from tools.preprocess_faust_partnete import fps_indices  # type: ignore
from tools.eval_cli import PatchToTextProj, build_model, load_ckpt  # type: ignore
from tools.dump_matching_patch_features import infer_proj_dims  # type: ignore
from models.pointnet2_utils import pc_normalize  # type: ignore


# -----------------------------
# I/O helpers

def _load_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    for k in ('points', 'xyz', 'data'):
        if k in data:
            arr = data[k]
            break
    else:
        # If the array is the only item
        if isinstance(data, np.lib.npyio.NpzFile) and len(list(data.keys())) == 1:
            arr = data[list(data.keys())[0]]
        else:
            raise ValueError(f"No 'points'/'xyz'/'data' key found in {path}")
    return np.asarray(arr)


def _load_npy(path: Path) -> np.ndarray:
    return np.asarray(np.load(str(path)))


def _load_txt(path: Path) -> np.ndarray:
    return np.asarray(np.loadtxt(str(path)))


def _load_mesh(path: Path) -> np.ndarray:
    """
    Load mesh file (PLY or GLB) and extract vertices.
    Uses trimesh (reuses pattern from precompute_partobjaverse_tiny.py).
    """
    import trimesh  # type: ignore
    obj = trimesh.load(str(path), force='mesh')
    if isinstance(obj, trimesh.Scene):
        # Merge all geometry to a single mesh in world coords (reuses existing pattern)
        if not obj.geometry:  # type: ignore[attr-defined]
            raise RuntimeError(f'No geometry in scene: {path}')
        mesh = trimesh.util.concatenate(tuple(obj.dump().geometry.values())) if hasattr(obj, 'dump') else trimesh.util.concatenate(tuple(obj.geometry.values()))  # type: ignore[attr-defined]
    else:
        mesh = obj
    if mesh.vertices is None or len(mesh.vertices) == 0:  # type: ignore[attr-defined]
        raise RuntimeError(f'No vertices found in mesh: {path}')
    return np.asarray(mesh.vertices, dtype=np.float32)  # type: ignore[attr-defined]


def load_point_cloud(p: Union[str, os.PathLike[str], io.IOBase]) -> np.ndarray:
    """Load a point cloud from *p* into an ``(N, C)`` float32 array.

    Supported formats: ``.npz``, ``.npy``, ``.txt`` / ``.xyz``, and
    ``.ply`` / ``.glb`` (the last two require *trimesh*).  File-like objects
    are treated as ``.npz``.

    Raises:
        FileNotFoundError: If *p* is a path that does not exist.
        ValueError: If the format is unsupported or the array has < 3 columns.
    """
    path: Path | io.IOBase
    if isinstance(p, io.IOBase):
        suf = '.npz'
        path = p
    else:
        path = Path(os.fspath(p))
        if not path.exists():
            raise FileNotFoundError(path)
        suf = path.suffix.lower()
    if suf == '.npz':
        arr = _load_npz(path)  # type: ignore[arg-type]
    elif suf == '.npy':
        arr = _load_npy(path)  # type: ignore[arg-type]
    elif suf in ('.txt', '.xyz'):
        arr = _load_txt(path)  # type: ignore[arg-type]
    elif suf in ('.ply', '.glb'):
        arr = _load_mesh(path)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported input format: {suf}")
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 3:
        raise ValueError("Point cloud must have at least 3 columns (x,y,z)")
    return arr.astype(np.float32, copy=False)


def resample_points(xyzf: np.ndarray, npoints: int, *, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    xyzf: (N,C) -> returns (N',C), (N',) indices into original rows.
    - If N > npoints: FPS (Farthest Point Sampling)
    - If N < npoints: sample with replacement
    - If N == npoints: identity

    Args:
        seed: Random seed for FPS.
    """
    N = xyzf.shape[0]
    if N == npoints:
        idx = np.arange(N, dtype=np.int64)
    elif N > npoints:
        # Use FPS for better spatial coverage (reuses existing code)
        idx = fps_indices(xyzf[:, :3], npoints, seed=seed)
    else:
        # If we have fewer points than needed, sample with replacement
        rng = np.random.RandomState(seed)
        extra = rng.choice(N, size=npoints - N, replace=True)
        idx = np.concatenate([np.arange(N, dtype=np.int64), extra], axis=0)
    return xyzf[idx], idx


def main(
    input: Union[str, os.PathLike[str], io.IOBase],
    ckpt: Optional[str] = None,
    output: Union[str, os.PathLike[str], io.IOBase] = '',
    arch: str = 'pointtransformer',
    num_group: int = 128,
    group_size: int = 32,
    npoints: int = 2048,
    device: Optional[str] = None,
    no_swap_yz: bool = False,
    no_normalize: bool = False,
    seed: int = 0,
    overwrite: bool = False,
    save_raw_only: bool = False
) -> None:
    """
    Extract patch features from a point cloud using a trained 2-stage model.

    Args:
        input: Path to point cloud (.npz/.npy/.txt/.xyz/.ply/.glb)
        ckpt: Path to model checkpoint. Defaults to model_ckpt/2stagemodel.pt
        output: Output file path (.npz or .pt) or file-like object. Defaults to <input>_patches.npz
        arch: Backbone architecture ('pointtransformer' or 'ppat')
        num_group: Number of patches (groups)
        group_size: Points per group (KNN)
        npoints: Number of points to sample from input
        device: Device ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'
        no_swap_yz: Disable the Y/Z axis swap
        seed: Sampling seed
        overwrite: Overwrite existing output file
        save_raw_only: Only save raw patch_emb; skip projection even if available
    """
    # Set defaults
    if ckpt is None:
        ckpt = str(_SEG_DIR / 'model_ckpt' / '2stagemodel.pt')
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_path: Path | io.IOBase
    if isinstance(output, io.IOBase):
        is_filelike = True
        out_path = output
    else:
        is_filelike = False
        input_str = str(input) if not isinstance(input, io.IOBase) else ''
        out_path = Path(os.fspath(output) if output else os.path.splitext(input_str)[0])
        if out_path.suffix.lower() not in ('.npz', '.pt'):
            out_path = Path(str(out_path) + '_patches.npz')
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path}. Pass --overwrite to replace.")

    # Load input
    pc = load_point_cloud(input)  # (N,C)
    C_in = pc.shape[1]

    # Normalize point cloud (center and scale to unit sphere)
    pc = pc_normalize(pc)

    # Resample to npoints using FPS for better spatial coverage
    pc_s, sample_to_input_idx = resample_points(pc, npoints, seed=seed)

    # Swap Y<->Z before feeding the model (matches training/eval)
    if not no_swap_yz and C_in >= 3:
        y = pc_s[:, 1].copy(); pc_s[:, 1] = pc_s[:, 2]; pc_s[:, 2] = y

    # Prepare tensor (B,C,N) with only XYZ by default
    pts = torch.from_numpy(pc_s[:, :3]).t().unsqueeze(0).contiguous().float()

    # Build model
    device_obj = torch.device(device)
    model = build_model(arch=arch, num_group=num_group, group_size=group_size).to(device_obj)

    # Decide projector output dim from checkpoint if available
    ckpt_path = Path(ckpt)
    _, proj_dim = infer_proj_dims(ckpt_path, default_in=384, default_out=512)
    proj = PatchToTextProj(in_dim=384, out_dim=proj_dim).to(device_obj)

    # Load checkpoint (eval_cli.load_ckpt requires non-None proj)
    st, _ = load_ckpt(model, proj, ckpt_path)
    model.eval();

    if 'proj' in st:
        proj.eval()

    with torch.no_grad():
        pts = pts.to(device_obj, non_blocking=True)
        # Forward to get patch embeddings, centers, and group membership indices
        patch_emb, patch_centers, patch_idx = model.forward_patches(pts)  # type: ignore[operator]
        # Shapes: (1,C,G), (1,3,G), (1,G,M)
        patch_emb = patch_emb.squeeze(0).transpose(0, 1).contiguous()     # (G,C)
        patch_centers = patch_centers.squeeze(0).transpose(0, 1).contiguous()  # (G,3)
        patch_idx = patch_idx.squeeze(0).contiguous()                     # (G,M)

        out = {
            'patch_emb': patch_emb.float().cpu().numpy(),
            'patch_centers': patch_centers.float().cpu().numpy(),
            'patch_indices': patch_idx.long().cpu().numpy(),
            'points_sampled': pc_s.astype(np.float32, copy=False),
            'sample_to_input_idx': sample_to_input_idx.astype(np.int64, copy=False),
            'meta_json': np.array([
                {
                    'source_path': str(input),
                    'ckpt_path': str(ckpt_path),
                    'arch': arch,
                    'num_group': int(num_group),
                    'group_size': int(group_size),
                    'npoints': int(npoints),
                    'swap_yz': (not no_swap_yz),
                    'device': str(device_obj),
                }
            ], dtype=object)
        }

        if 'proj' in st and not save_raw_only:
            feat = proj(patch_emb.transpose(0, 1).unsqueeze(0))   # (1,G,D)
            feat = feat.squeeze(0).contiguous()                   # (G,D)
            out['patch_feat'] = feat.float().cpu().numpy()

    # Save
    if is_filelike:
        np.savez(output, **out)
        print(f"Saved patches to file-like object")
    else:
        assert isinstance(out_path, Path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == '.npz':
            np.savez_compressed(out_path, **out)
        else:
            torch.save(out, out_path)
        print(f"Saved patches to {out_path}")
    print("  - patch_emb:", out['patch_emb'].shape)
    if 'patch_feat' in out:
        print("  - patch_feat:", out['patch_feat'].shape)
    print("  - patch_centers:", out['patch_centers'].shape)
    print("  - patch_indices:", out['patch_indices'].shape)


if __name__ == '__main__':
    fire.Fire(main)
