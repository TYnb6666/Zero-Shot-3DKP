"""Furthest Point Sampling (FPS) on CPU using numpy.

Provides a pure-numpy implementation of greedy FPS that iteratively selects
the point farthest from the already-chosen set until *k* points are picked.
Used by :pyfunc:`inference.extract_patch_features.resample_points` to
downsample raw point clouds before feeding them to the patch encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray


def fps_indices(xyz: NDArray[np.floating], k: int, *, seed: int = 0) -> NDArray[np.int64]:
    """Select *k* points from *xyz* via Furthest Point Sampling.

    Starting from a random seed point, each subsequent point is chosen as
    the one whose minimum squared-distance to *all* previously selected
    points is largest.  Runs in O(N * k) time.

    Args:
        xyz: Point cloud of shape ``(N, 3)``.
        k: Number of points to sample.
        seed: Random seed used to pick the initial point.

    Returns:
        Integer index array of shape ``(k,)`` into *xyz*.
        If ``N <= k`` the full ``np.arange(N)`` is returned.
    """
    N: int = xyz.shape[0]
    if N <= k:
        return np.arange(N, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idxs = np.empty(k, dtype=np.int64)
    idxs[0] = int(rng.randint(0, N))
    dists: NDArray[np.float64] = np.full(N, np.inf, dtype=np.float64)
    last = xyz[idxs[0]][None, :]
    dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    for i in range(1, k):
        idxs[i] = int(np.argmax(dists))
        last = xyz[idxs[i]][None, :]
        dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    return idxs


def maybe_presample(xyz: np.ndarray, labels: np.ndarray, *, max_input: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """If xyz too large, randomly pre-sample to max_input before FPS. Returns (xyz_sub, labels_sub, idx_map).
    idx_map maps indices into xyz_sub back to original xyz indices.
    """
    N = xyz.shape[0]
    if N <= max_input:
        idx_map = np.arange(N, dtype=np.int64)
        return xyz, labels, idx_map
    rng = np.random.RandomState(seed)
    sel = rng.choice(N, size=max_input, replace=False)
    sel.sort()
    return xyz[sel], labels[sel], sel


def load_obj_vertices(obj_path: Path) -> np.ndarray:
    """Simple .obj vertex reader, returns (N,3) float32. Ignores faces."""
    verts = []
    with open(obj_path, 'r') as f:
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
        raise RuntimeError(f"No vertices found in {obj_path}")
    return np.asarray(verts, dtype=np.float32)


def process_faust(root: Path, *, granularity: str, npoints: int, max_input: int, seed: int, overwrite: bool):
    assert granularity in ('coarse', 'fine')
    label_json = root / ('coarse_gt.json' if granularity == 'coarse' else 'fine_grained_gt.json')
    scans_dir = root / 'scans'
    if not label_json.exists():
        raise FileNotFoundError(f"FAUST labels not found: {label_json}")
    with open(label_json, 'r') as f:
        labels_all = json.load(f)
    # Collect label names across all scans for consistent indexing
    all_names = set()
    for arr in labels_all.values():
        all_names.update(arr)
    label_names = sorted(all_names)
    name_to_idx = {n: i for i, n in enumerate(label_names)}

    for key, name_list in labels_all.items():
        obj_path = scans_dir / f'{key}.obj'
        if not obj_path.exists():
            print(f"[FAUST] missing OBJ for {key}, skipping")
            continue
        out_path = obj_path.with_name(f'{key}_pts{npoints}.npz')
        if out_path.exists() and not overwrite:
            continue
        xyz = load_obj_vertices(obj_path)
        labs_str = np.asarray(name_list, dtype=object)
        if labs_str.shape[0] != xyz.shape[0]:
            print(f"[FAUST] label length mismatch for {key}: verts={xyz.shape[0]} labels={labs_str.shape[0]}")
            m = min(labs_str.shape[0], xyz.shape[0])
            xyz = xyz[:m]
            labs_str = labs_str[:m]
        labs_idx = np.vectorize(name_to_idx.get)(labs_str)
        # pre-sample then FPS
        xyz_sub, labs_sub, idx_map = maybe_presample(xyz, labs_idx, max_input=max_input, seed=seed)
        fps_idx = fps_indices(xyz_sub, npoints, seed=seed)
        pts = xyz_sub[fps_idx]
        lbl = labs_sub[fps_idx]
        # Map sampled points back to original vertex indices in the OBJ
        # idx_map maps indices into xyz_sub back to original xyz indices
        try:
            orig_vertex_index = idx_map[fps_idx].astype(np.int64)
        except Exception:
            # Fallback if idx_map is identity
            orig_vertex_index = fps_idx.astype(np.int64)
        lbl_str = np.asarray([label_names[i] if (0 <= i < len(label_names)) else 'unknown' for i in lbl], dtype=object)
        np.savez_compressed(
            out_path,
            points=pts.astype(np.float32),
            labels=lbl.astype(np.int64),
            labels_str=lbl_str,
            label_names=np.asarray(label_names, dtype=object),
            category='human',
            slug=key,
            orig_vertex_index=orig_vertex_index,
        )
        print(f"[FAUST] wrote {out_path}")


def read_ply_points(path: Path) -> np.ndarray:
    """Read PLY vertices as (N,3) float32. Requires trimesh or plyfile."""
    try:
        import trimesh
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Trimesh):
            return np.asarray(loaded.vertices, dtype=np.float32)
        if hasattr(loaded, 'vertices'):
            return np.asarray(loaded.vertices, dtype=np.float32)  # type: ignore[union-attr]
        raise RuntimeError('Unsupported PLY structure')
    except Exception:
        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(path))
            v = plydata['vertex']
            xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
            return xyz
        except Exception as e:
            raise RuntimeError(f"Failed to read PLY: {path} ({e})")


def process_partnete(partslip_root: Path, split_root: Optional[Path], *, npoints: int, max_input: int, seed: int, overwrite: bool):
    # part names
    meta_path = partslip_root / 'PartNetE_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"PartNetE_meta.json not found under {partslip_root}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    # default split root: '<root>/data/test' if extracted directory exists
    if split_root is None:
        # attempt to locate extracted 'test' directory next to the zip
        cand = partslip_root / 'data' / 'test'
        if not cand.exists():
            raise FileNotFoundError('Please provide --partnete_split_root pointing to extracted test folder (contains category dirs).')
        split_root = cand
    # iterate categories
    for cat_dir in sorted([p for p in split_root.iterdir() if p.is_dir()]):
        category = cat_dir.name
        label_names = meta.get(category, None)
        if label_names is None:
            print(f"[PartNet-E] Warning: category {category} not in meta; using empty label_names")
            label_names = []
        for inst_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
            slug = inst_dir.name
            ply_path = inst_dir / 'pc.ply'
            label_path = inst_dir / 'label.npy'
            out_path = inst_dir / f'pc_{npoints}.npz'
            if out_path.exists() and not overwrite:
                continue
            if not (ply_path.exists() and label_path.exists()):
                print(f"[PartNet-E] missing files under {inst_dir}, skipping")
                continue
            # read points
            xyz = read_ply_points(ply_path)
            # read labels dict
            try:
                arr = np.load(label_path, allow_pickle=True)
                lab_dict = arr.item() if isinstance(arr, np.ndarray) else arr
                labs = np.asarray(lab_dict.get('semantic_seg', None))
                if labs is None:
                    raise RuntimeError('semantic_seg not found')
            except Exception as e:
                print(f"[PartNet-E] failed to read labels: {label_path} ({e})")
                continue
            if xyz.shape[0] != labs.shape[0]:
                m = min(xyz.shape[0], labs.shape[0])
                xyz = xyz[:m]
                labs = labs[:m]
            # pre-sample then FPS
            xyz_sub, labs_sub, _idx_map = maybe_presample(xyz, labs, max_input=max_input, seed=seed)
            fps_idx = fps_indices(xyz_sub, npoints, seed=seed)
            pts = xyz_sub[fps_idx]
            lbl = labs_sub[fps_idx].astype(np.int64)
            # map to names if possible; mark any out-of-range or negative as -1 (unknown)
            if label_names:
                unknown_mask = (lbl < 0) | (lbl >= len(label_names))
                if unknown_mask.any():
                    lbl = lbl.copy()
                    lbl[unknown_mask] = -1
                lbl_str = np.empty(lbl.shape[0], dtype=object)
                valid = lbl >= 0
                lbl_str[valid] = [label_names[int(i)] for i in lbl[valid]]
                lbl_str[~valid] = 'unknown'
            else:
                lbl_str = np.array(['unknown'] * lbl.shape[0], dtype=object)
            np.savez_compressed(out_path, points=pts.astype(np.float32), labels=lbl, labels_str=lbl_str,
                                label_names=np.asarray(label_names, dtype=object), category=category, slug=slug)
            print(f"[PartNet-E] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser('Preprocess FAUST and PartNet-E to fixed-size point clouds with labels')
    ap.add_argument('--faust_root', type=str, default='', help='Path to faust/SATR_DATA')
    ap.add_argument('--faust_granularity', type=str, default='coarse', choices=['coarse','fine'])
    ap.add_argument('--partnete_root', type=str, default='', help='Path to PartSLIP repo root (contains PartNetE_meta.json)')
    ap.add_argument('--partnete_split_root', type=str, default='', help='Extracted split root (e.g., data/test). If empty, tries <root>/data/test')
    ap.add_argument('--npoints', type=int, default=2048)
    ap.add_argument('--max_input', type=int, default=10000, help='If point count exceeds this, randomly pre-sample before FPS')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true', default=False)
    args = ap.parse_args()

    if args.faust_root:
        process_faust(Path(args.faust_root), granularity=args.faust_granularity, npoints=args.npoints,
                      max_input=args.max_input, seed=args.seed, overwrite=args.overwrite)
    if args.partnete_root:
        split_root = Path(args.partnete_split_root) if args.partnete_split_root else None
        process_partnete(Path(args.partnete_root), split_root, npoints=args.npoints, max_input=args.max_input,
                         seed=args.seed, overwrite=args.overwrite)
    if not args.faust_root and not args.partnete_root:
        print('Nothing to do: provide --faust_root and/or --partnete_root')


if __name__ == '__main__':
    main()
