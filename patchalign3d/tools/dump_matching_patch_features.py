"""Checkpoint inspection utilities for the patch feature pipeline.

Provides :func:`infer_proj_dims` which peeks inside a saved checkpoint to
determine the input/output dimensions of the ``PatchToTextProj`` linear
layer, so the caller can construct a matching module before loading
weights.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from tools.eval_cli import PatchToTextProj, build_model, load_ckpt  # type: ignore


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


def load_obj_vertices(path: Path) -> np.ndarray:
    """Simple .obj vertex reader, returns (N,3) float32. Ignores faces."""
    verts = []
    with open(path, 'r') as f:
        for line in f:
            if not line:
                continue
            if line.startswith('v '):
                parts = line.strip().split()
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    verts.append((x, y, z))
                except Exception:
                    continue
    if not verts:
        raise RuntimeError(f"No vertices found in {path}")
    return np.asarray(verts, dtype=np.float32)


def read_ply_points(path: Path) -> np.ndarray:
    """Read PLY vertices as (N,3) float32. Requires trimesh or plyfile."""
    try:
        import trimesh  # type: ignore
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Trimesh):
            return np.asarray(loaded.vertices, dtype=np.float32)
        if hasattr(loaded, 'vertices'):
            return np.asarray(loaded.vertices, dtype=np.float32)  # type: ignore[union-attr]
        raise RuntimeError('Unsupported PLY structure')
    except Exception:
        try:
            from plyfile import PlyData  # type: ignore
            plydata = PlyData.read(str(path))
            v = plydata['vertex']
            xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
            return xyz
        except Exception as e:
            raise RuntimeError(f"Failed to read PLY: {path} ({e})")


def read_off_vertices(path: Path) -> np.ndarray:
    """Read OFF mesh vertices as (N,3) float32."""
    try:
        import trimesh  # type: ignore
        m = trimesh.load(path, process=False)
        if isinstance(m, trimesh.Trimesh) and hasattr(m, 'vertices'):
            v = np.asarray(m.vertices, dtype=np.float32)
            if v.ndim == 2 and v.shape[1] >= 3:
                return v[:, :3].astype(np.float32)
    except Exception:
        pass
    # Fallback minimal parser
    with open(path, 'r') as f:
        # Read first non-empty line
        first = ''
        while True:
            first = f.readline()
            if not first:
                raise RuntimeError(f'Empty OFF file: {path}')
            first = first.strip()
            if first and not first.startswith('#'):
                break
        if not (first == 'OFF' or first.startswith('OFF') or first == 'COFF'):
            # Some files may include counts on the same line; try to strip leading token
            if 'OFF' not in first:
                raise RuntimeError(f'Invalid OFF header in {path}: {first[:32]}')
        # Read counts (n_verts n_faces n_edges)
        nverts = None
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f'Missing counts line in {path}')
            line = line.strip()
            if (not line) or line.startswith('#'):
                continue
            parts = line.split()
            try:
                nverts = int(parts[0]); _ = int(parts[1])
                break
            except Exception:
                continue
        verts = []
        for _ in range(int(nverts)):  # type: ignore[arg-type]
            line = f.readline()
            while line and (line.strip().startswith('#') or len(line.strip()) == 0):
                line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                verts.append((x, y, z))
            except Exception:
                continue
    if not verts:
        raise RuntimeError(f'No vertices parsed from {path}')
    return np.asarray(verts, dtype=np.float32)


def fps_indices(xyz: np.ndarray, k: int, *, seed: int = 0) -> np.ndarray:
    """Simple numpy Furthest Point Sampling. Returns indices of size k."""
    N = int(xyz.shape[0])
    if N <= k:
        return np.arange(N, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idxs = np.empty(k, dtype=np.int64)
    idxs[0] = int(rng.randint(0, N))
    dists = np.full(N, np.inf, dtype=np.float64)
    last = xyz[idxs[0]][None, :]
    dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    for i in range(1, k):
        idxs[i] = int(np.argmax(dists))
        last = xyz[idxs[i]][None, :]
        dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    return idxs


def read_points_from_mesh(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == '.obj':
        return load_obj_vertices(path)
    elif ext == '.ply':
        return read_ply_points(path)
    elif ext == '.off':
        return read_off_vertices(path)
    else:
        raise RuntimeError(f'Unsupported mesh extension: {ext} ({path})')


def gather_mesh_files(roots: List[str], *, exts: Tuple[str, ...] = ('.obj', '.ply', '.off')) -> List[Path]:
    files: List[Path] = []
    import glob as _glob
    for r in roots:
        # Expand globs if any
        if any(ch in r for ch in ['*', '?', '[']):
            for p in _glob.glob(r):
                bp = Path(p)
                if bp.is_file() and bp.suffix.lower() in exts:
                    files.append(bp)
                elif bp.is_dir():
                    for ext in exts:
                        files.extend(bp.rglob(f'*{ext}'))
            continue
        base = Path(r)
        if base.is_file():
            if base.suffix.lower() in exts:
                files.append(base)
            continue
        if base.is_dir():
            for ext in exts:
                files.extend(base.rglob(f'*{ext}'))
    # Deduplicate while preserving order
    seen = set(); out = []
    for f in files:
        if f not in seen:
            out.append(f); seen.add(f)
    return out


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    c = pts.mean(axis=0, keepdims=True)
    p = pts - c
    r = (p ** 2).sum(axis=1) ** 0.5
    m = float(r.max()) if r.size > 0 else 1.0
    if m > 0:
        p = p / m
    # swap Y/Z
    p = p.copy(); p[:, [1, 2]] = p[:, [2, 1]]
    return p, c.reshape(-1), m


@torch.no_grad()
def run_model_for_points(points_bcn: torch.Tensor, *, device: torch.device, arch: str,
                         num_group: int, group_size: int, ckpt_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Forward one batch (B,3,N) through model+proj; returns (patch_feat, patch_centers, membership, preproj, proj_dims)."""
    # Infer projector dims to construct matching module before loading
    proj_in, proj_out = infer_proj_dims(ckpt_path)
    model = build_model(arch=arch, num_group=num_group, group_size=group_size).to(device)
    proj = PatchToTextProj(in_dim=int(proj_in), out_dim=int(proj_out)).to(device)
    _ = load_ckpt(model, proj, str(ckpt_path))
    model.eval(); proj.eval()

    pe, pc, pi = model.forward_patches(points_bcn)  # type: ignore[operator]  # pe: (B,C,G), pc: (B,3,G), pi: (B,G,M)
    feat = proj(pe)  # (B,G,D), L2 normalized in module
    # Move to CPU numpy
    feat_cpu = feat.detach().cpu().numpy()
    pc_cpu = pc.detach().cpu().numpy().transpose(0, 2, 1)  # (B,G,3)
    pi_cpu = pi.detach().cpu().numpy()  # (B,G,M)
    preproj_cpu = pe.detach().cpu().numpy().transpose(0, 2, 1)  # (B,G,C)
    return feat_cpu, pc_cpu, pi_cpu, preproj_cpu, (int(proj_in), int(proj_out))


def main():
    ap = argparse.ArgumentParser('Dump dual-model patch features for mesh datasets')
    # Allow passing multiple roots per occurrence and/or repeating the flag; supports globs
    ap.add_argument('--roots', type=str, nargs='+', action='append', required=True,
                    help='Root directory(ies) (repeatable). You can pass multiple paths after --roots and/or use globs.')
    ap.add_argument('--roots_file', type=str, default='', help='Optional file with one root path per line')
    ap.add_argument('--out_dir', type=str, required=True, help='Directory to write NPZ dumps')
    ap.add_argument('--flat', action='store_true', default=False, help='If set, flattens output (no directory mirroring)')
    ap.add_argument('--stage1_ckpt', type=str, required=True)
    ap.add_argument('--stage2_ckpt', type=str, required=True)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--npoints', type=int, default=2048)
    ap.add_argument('--max_input', type=int, default=200000, help='If more than this many input points, randomly subsample before FPS')
    ap.add_argument('--num_group', type=int, default=128)
    ap.add_argument('--group_size', type=int, default=32)
    ap.add_argument('--arch', type=str, default='pointtransformer', choices=['pointtransformer', 'ppat'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='Optional limit on number of meshes to process')
    ap.add_argument('--overwrite', action='store_true', default=False)
    ap.add_argument('--only_off', action='store_true', default=False, help='Process only .off files')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Flatten roots from list-of-lists
    roots: List[str] = []
    for group in (args.roots or []):
        roots.extend(list(group))
    if args.roots_file:
        from pathlib import Path as _P
        try:
            for ln in _P(args.roots_file).read_text().splitlines():
                ln = ln.strip()
                if ln:
                    roots.append(ln)
        except Exception:
            pass
    # Select extensions based on filter
    exts = ('.off',) if bool(args.only_off) else ('.obj', '.ply', '.off')
    files = gather_mesh_files(roots, exts=exts)
    # Build list of realized base directories (expand globs) to mirror structure correctly
    import glob as _glob
    base_dirs: List[Path] = []
    for r in roots:
        if any(ch in r for ch in ['*', '?', '[']):
            for p in _glob.glob(r):
                bp = Path(p)
                if bp.is_dir():
                    base_dirs.append(bp)
                elif bp.is_file():
                    base_dirs.append(bp.parent)
        else:
            bp = Path(r)
            if bp.is_dir():
                base_dirs.append(bp)
            elif bp.is_file():
                base_dirs.append(bp.parent)
    # Deduplicate and sort by depth descending so more specific roots match first
    def _depth(p: Path) -> int:
        try:
            return len(p.resolve().parts)
        except Exception:
            return len(p.parts)
    _seen = set()
    _uniq: List[Path] = []
    for b in base_dirs:
        try:
            key = str(b.resolve())
        except Exception:
            key = str(b)
        if key in _seen:
            continue
        _seen.add(key)
        _uniq.append(b)
    base_dirs = sorted(_uniq, key=_depth, reverse=True)
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm  # type: ignore
    it = tqdm(files, desc='Meshes', smoothing=0.9)
    for f in it:
        try:
            pts_all = read_points_from_mesh(f)
        except Exception as e:
            it.write(f"[skip] failed to read {f}: {e}")
            continue
        if not isinstance(pts_all, np.ndarray) or pts_all.ndim != 2 or pts_all.shape[1] != 3:
            it.write(f"[skip] invalid points shape for {f}: {None if not isinstance(pts_all, np.ndarray) else pts_all.shape}")
            continue
        N = int(pts_all.shape[0])
        if N == 0:
            it.write(f"[skip] empty points for {f}")
            continue

        # Optionally pre-subsample before FPS for speed
        if N > int(args.max_input):
            sel = np.random.choice(N, size=int(args.max_input), replace=False)
            sel.sort(); pts_all = pts_all[sel]
        # FPS to target npoints
        if int(args.npoints) > 0:
            fps_idx = fps_indices(pts_all, int(args.npoints), seed=int(args.seed))
            pts = pts_all[fps_idx]
            orig_input_index = fps_idx.astype(np.int64)
        else:
            pts = pts_all
            orig_input_index = np.arange(pts.shape[0], dtype=np.int64)

        # Save original-space points
        points_orig = pts.copy()
        # Normalize + swap
        points_norm, norm_center, norm_scale = normalize_points(points_orig)

        # Prepare model input (B,3,N)
        pts_bcn = torch.from_numpy(points_norm[None, :, :].astype(np.float32)).to(device).transpose(2, 1).contiguous()

        # Stage 1
        feat1, pc1, pi1, pre1, proj_dims1 = run_model_for_points(pts_bcn, device=device, arch=str(args.arch),
                                                                  num_group=int(args.num_group), group_size=int(args.group_size),
                                                                  ckpt_path=Path(args.stage1_ckpt))
        # Stage 2
        feat2, pc2, pi2, pre2, proj_dims2 = run_model_for_points(pts_bcn, device=device, arch=str(args.arch),
                                                                  num_group=int(args.num_group), group_size=int(args.group_size),
                                                                  ckpt_path=Path(args.stage2_ckpt))

        # Extract batch dim (B=1)
        feat1 = feat1[0]; feat2 = feat2[0]
        pc1 = pc1[0]; pc2 = pc2[0]
        pi1 = pi1[0]; pi2 = pi2[0]
        pre1 = pre1[0]; pre2 = pre2[0]

        # Nearest assignment of points to each stage's patch centers (normalized frame)
        def nearest_assign(pts_: np.ndarray, centers_: np.ndarray) -> np.ndarray:
            # pts_: (N,3), centers_: (G,3)
            # cKDTree for speed on large N
            try:
                from scipy.spatial import cKDTree  # type: ignore[attr-defined]
                tree = cKDTree(centers_)
                _, idx = tree.query(pts_, k=1)
                return idx.astype(np.int64)
            except Exception:
                d = np.linalg.norm(pts_[:, None, :] - centers_[None, :, :], axis=2)
                return d.argmin(axis=1).astype(np.int64)

        pt2patch1 = nearest_assign(points_norm, pc1)
        pt2patch2 = nearest_assign(points_norm, pc2)

        # Map centers back to original space (invert swap and normalization)
        pc1_unswapped = pc1[:, [0, 2, 1]]
        pc2_unswapped = pc2[:, [0, 2, 1]]
        pc1_orig = pc1_unswapped * float(norm_scale) + norm_center[None, :]
        pc2_orig = pc2_unswapped * float(norm_scale) + norm_center[None, :]

        # Alignment mapping between stage2 and stage1 patches (nearest center in normalized frame)
        try:
            from scipy.spatial import cKDTree  # type: ignore[attr-defined]
            t1 = cKDTree(pc1)
            _, map2to1 = t1.query(pc2, k=1)
            map2to1 = map2to1.astype(np.int64)
        except Exception:
            d = np.linalg.norm(pc2[:, None, :] - pc1[None, :, :], axis=2)
            map2to1 = d.argmin(axis=1).astype(np.int64)

        # Prepare output path
        matched_root = None
        matched_rel = None
        for base in base_dirs:
            try:
                rel = Path(f).relative_to(base)
                matched_root = base
                matched_rel = rel
                break
            except Exception:
                continue
        dataset_name = matched_root.name if matched_root is not None else f.parent.name
        if not bool(args.flat) and matched_rel is not None:
            out_dir_final = out_root / dataset_name / matched_rel.parent
            out_dir_final.mkdir(parents=True, exist_ok=True)
            out_path = out_dir_final / (f.stem + '.npz')
            slug = f"{dataset_name}/{matched_rel.as_posix()}"
        else:
            # Flattened layout
            slug = (matched_rel.as_posix() if matched_rel is not None else f.name)
            safe_slug = f"{dataset_name}__" + slug.replace('\\', '__').replace('/', '__')
            out_path = out_root / f'{safe_slug}.npz'
        if out_path.exists() and not args.overwrite:
            continue

        out = {
            # Input and normalization
            'points_orig': points_orig.astype(np.float32),
            'points': points_norm.astype(np.float32),  # normalized + swapped
            'orig_input_index': orig_input_index.astype(np.int64),
            'norm_center': norm_center.astype(np.float32),
            'norm_scale': np.float32(norm_scale),
            'axis_swap': np.array([0, 2, 1], dtype=np.int64),
            # Stage 1
            'patch_centers_stage1': pc1.astype(np.float32),
            'patch_centers_orig_stage1': pc1_orig.astype(np.float32),
            'patch_membership_idx_stage1': pi1.astype(np.int64),
            'point_to_patch_nearest_stage1': pt2patch1.astype(np.int64),
            'patch_feat_stage1': feat1.astype(np.float32),
            'patch_feat_preproj_stage1': pre1.astype(np.float32),
            # Stage 2
            'patch_centers_stage2': pc2.astype(np.float32),
            'patch_centers_orig_stage2': pc2_orig.astype(np.float32),
            'patch_membership_idx_stage2': pi2.astype(np.int64),
            'point_to_patch_nearest_stage2': pt2patch2.astype(np.int64),
            'patch_feat_stage2': feat2.astype(np.float32),
            'patch_feat_preproj_stage2': pre2.astype(np.float32),
            # Alignment mapping
            'stage2_to_stage1_nearest': map2to1.astype(np.int64),
            # Meta
            'stage1_ckpt': str(args.stage1_ckpt),
            'stage2_ckpt': str(args.stage2_ckpt),
            'source': str(f),
            'slug': slug,
            'proj_dims_stage1': np.array(proj_dims1, dtype=np.int64),
            'proj_dims_stage2': np.array(proj_dims2, dtype=np.int64),
        }
        np.savez_compressed(out_path, **out)
        it.set_postfix_str(f"saved -> {out_path.name}")


if __name__ == '__main__':
    main()
