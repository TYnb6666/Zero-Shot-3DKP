"""Model building, checkpoint loading, and CLIP text-encoding utilities.

This module supplies the core inference building-blocks shared by the
patch-feature extraction and exploration pipelines:

* :class:`PatchToTextProj` — lightweight linear projector from patch
  embeddings to the CLIP text-embedding space.
* :func:`build_model` — construct a PointTransformer (or PPAT) encoder
  from a simple config dict.
* :func:`load_ckpt` — load a checkpoint dict into an encoder + projector
  pair with tolerant (non-strict) state-dict matching.
* :class:`LRUTextCache` — LRU cache that lazily encodes part-name strings
  into CLIP text features using configurable prompt templates.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
import glob as _glob
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make the patchalign3d model modules importable so that dynamic
# ``__import__('models.PointTransformer_patched', ...)`` calls succeed.
# ---------------------------------------------------------------------------
import sys as _sys

_THIS = Path(__file__).resolve()
_SEG_DIR = _THIS.parent.parent          # .../patchalign3d
_ROOT = _SEG_DIR.parent                 # repo root
_sys.path.insert(0, str(_ROOT))
_sys.path.insert(0, str(_SEG_DIR))
_sys.path.insert(0, str(_SEG_DIR / 'models'))


# ── Model & projector ─────────────────────────────────────────────────────

class PatchToTextProj(nn.Module):
    """Project raw patch embeddings into the CLIP text-feature space.

    A single linear layer followed by L2 normalization.  Input shape is
    ``(B, C, G)`` (batch, channels, groups) and output is ``(B, G, D)``
    where *D* equals *out_dim*.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: transpose, project, L2-normalize."""
        x = patch_emb.transpose(1, 2)          # (B,C,G) → (B,G,C)
        x = self.proj(x)                        # (B,G,C) → (B,G,D)
        x = F.normalize(x, dim=-1)              # unit-norm along D
        return x


def build_model(
    arch: str = 'pointtransformer',
    *,
    num_group: int = 128,
    group_size: int = 32,
    use_color: bool = False,
    use_normal: bool = False,
    trans_dim: int = 384,
) -> nn.Module:
    """Construct a point-cloud patch encoder.

    Args:
        arch: Architecture name — ``'pointtransformer'`` or ``'ppat'``.
        num_group: Number of point groups (patches).
        group_size: Points per group (KNN neighbourhood size).
        use_color: Include per-point RGB channels.
        use_normal: Include per-point surface normals.
        trans_dim: Transformer hidden dimension.

    Returns:
        An ``nn.Module`` with a ``forward_patches`` method.
    """
    from easydict import EasyDict  # type: ignore[import-untyped]

    cfg = EasyDict(
        trans_dim=trans_dim, depth=12, drop_path_rate=0.1, cls_dim=50,
        num_heads=6, group_size=group_size, num_group=num_group,
        encoder_dims=256, color=use_color, num_classes=16,
    )
    if arch == 'pointtransformer':
        MODULE = __import__('models.PointTransformer_patched', fromlist=['get_model'])
        return MODULE.get_model(cfg)  # type: ignore[no-any-return]
    if arch == 'ppat':
        cfg.update({
            'patches': num_group, 'patch_radius': 0.2, 'nsample': group_size,
            'in_channel': (3 if use_normal else 0) + (3 if use_color else 0),
            'mlp_dim': trans_dim * 4, 'sa_dim': 128, 'rel_pe': False,
        })
        MODULE = __import__('models.ppat', fromlist=['get_model'])
        return MODULE.get_model(cfg)  # type: ignore[no-any-return]
    raise ValueError(f"Unknown arch: {arch}")


def load_ckpt(
    model: nn.Module,
    proj: PatchToTextProj,
    path: str | Path,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a checkpoint into *model* and *proj* (non-strict).

    Args:
        model: The patch encoder module.
        proj: The :class:`PatchToTextProj` projector.
        path: Filesystem path to a ``.pt`` checkpoint.
        verbose: Print loading diagnostics to stdout.

    Returns:
        ``(state_dict, logs)`` where *state_dict* is the raw checkpoint
        dict and *logs* records any missing / unexpected keys.
    """
    st: Dict[str, Any] = torch.load(str(path), map_location='cpu')
    logs: Dict[str, Any] = {
        'ckpt_keys': list(st.keys()),
        'model_missing': [], 'model_unexpected': [],
        'proj_missing': [], 'proj_unexpected': [],
    }
    if 'model' in st:
        try:
            missing, unexpected = model.load_state_dict(st['model'], strict=False)
            logs['model_missing'] = missing
            logs['model_unexpected'] = unexpected
        except Exception as e:
            logs['model_error'] = str(e)
    else:
        logs['model_info'] = 'no model key in checkpoint'
    if 'proj' in st:
        try:
            missing, unexpected = proj.load_state_dict(st['proj'], strict=False)
            logs['proj_missing'] = missing
            logs['proj_unexpected'] = unexpected
        except Exception as e:
            logs['proj_error'] = str(e)
    else:
        logs['proj_info'] = 'no proj key in checkpoint'
    if verbose:
        print('[ckpt] keys:', logs.get('ckpt_keys', [])[:6], '...')
        pm = logs.get('model_missing', [])
        pu = logs.get('model_unexpected', [])
        print(f"[ckpt] model load: missing={len(pm)} unexpected={len(pu)}")
        pm = logs.get('proj_missing', [])
        pu = logs.get('proj_unexpected', [])
        print(f"[ckpt] proj load: missing={len(pm)} unexpected={len(pu)}")
        if hasattr(proj, 'proj'):
            try:
                print('[ckpt] proj weight shape:', tuple(proj.proj.weight.shape))
            except Exception:
                pass
    return st, logs


# ── Text encoding ─────────────────────────────────────────────────────────

def _clean_text(s: str) -> str:
    """Normalize a part-name string for caching and template expansion.

    Lower-cases, replaces underscores with spaces, strips punctuation,
    and collapses whitespace.
    """
    s = s.strip().lower().replace('_', ' ')
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class LRUTextCache:
    """LRU cache that encodes part-name strings into CLIP text features.

    Given a CLIP model and tokenizer, this class lazily computes the
    mean-pooled text embedding for each ``(name, category, setting)``
    triple using a set of prompt templates (e.g. ``"a {} of a {}"``).
    Results are cached in an :class:`~collections.OrderedDict` with LRU
    eviction once *capacity* is exceeded.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        backend: str = 'clip',
        capacity: int = 20_000,
        clip_model: Any = None,
        tokenizer: Any = None,
        text_dim: int = 512,
    ) -> None:
        self.backend = backend
        self.device = device
        self.capacity = int(capacity)
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.text_dim = int(text_dim)
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _touch(self, key: str) -> None:
        """Move *key* to the most-recently-used end of the cache."""
        try:
            self._store.move_to_end(key)
        except KeyError:
            pass

    @torch.no_grad()
    def encode_label_for_sample(
        self,
        name: str,
        category: Optional[str],
        setting: str,
        initial_texts: Sequence[str] = (),
    ) -> torch.Tensor:
        """Encode a single part name into a CLIP text feature vector.

        Args:
            name: Part name (e.g. ``"wheel"``).
            category: Optional object category (e.g. ``"car"``).
            setting: Template setting -- ``'part_only'``,
                ``'part_plus_cat'``, or ``'ensemble'``.
            initial_texts: Additional prompt strings prepended to the
                template-expanded list.

        Returns:
            A 1-D tensor of shape ``(text_dim,)`` on *self.device*.
        """
        name_k = _clean_text(name)
        cat_k = _clean_text(category) if category else ''
        cache_key = f"{name_k}||{cat_k}||{setting}"
        if cache_key in self._store:
            self._touch(cache_key)
            return self._store[cache_key]

        texts: List[str] = list(initial_texts)
        PART_ONLY_TEMPLATES = ["{}", "a {}", "{} part"]
        PART_PLUS_CAT_TEMPLATES = [
            "a {} of a {}", "the {} of a {}", "{} of {}", "a {} part of a {}",
        ]
        if setting in ('part_plus_cat', 'ensemble') and cat_k:
            for tpl in PART_PLUS_CAT_TEMPLATES:
                slots = tpl.count("{}")
                if slots == 2:
                    texts.append(tpl.format(name_k, cat_k))
                elif slots == 1:
                    texts.append(tpl.format(f"{cat_k} {name_k}"))
                else:
                    texts.append(f"{cat_k} {name_k}")
        if setting in ('part_only', 'ensemble') or not cat_k:
            for tpl in PART_ONLY_TEMPLATES:
                texts.append(tpl.format(name_k) if tpl.count("{}") == 1 else name_k)

        toks = self.tokenizer(texts).to(self.device)
        feat = F.normalize(self.clip_model.encode_text(toks), dim=-1)
        out = (
            F.normalize(feat.mean(dim=0, keepdim=True), dim=-1)
            .squeeze(0)
            .float()
            .to(self.device)
        )
        self._store[cache_key] = out
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return out

    def encode_labels_for_sample(
        self,
        names: List[str],
        category: Optional[str],
        setting: str,
    ) -> torch.Tensor:
        """Encode multiple part names and stack into a ``(K, text_dim)`` tensor."""
        if not names:
            return torch.empty(0, self.text_dim, device=self.device)
        vecs = [self.encode_label_for_sample(n, category, setting) for n in names]
        return torch.stack(vecs, dim=0)


class BankTextCache:
    def __init__(self, *, device: torch.device, bank_part_only: Optional[LoadedTextBank], bank_part_cat: Optional[LoadedTextBank],
                 fallback: Optional[LRUTextCache] = None, strict: bool = False):
        self.device = device
        self.po = bank_part_only
        self.pc = bank_part_cat
        self.fallback = fallback
        self.strict = bool(strict)
        dims = [b.dim for b in [self.po, self.pc] if b is not None]
        self.text_dim = int(dims[0]) if dims else (fallback.text_dim if fallback is not None else 512)

    def _lookup_po(self, name: str) -> Optional[torch.Tensor]:
        if self.po is None:
            return None
        k = _clean_text(name)
        idx = self.po.key_to_idx.get(k)
        if idx is None:
            return None
        return F.normalize(self.po.emb[idx].to(self.device), dim=-1)

    def _lookup_pc(self, name: str, category: Optional[str]) -> Optional[torch.Tensor]:
        if self.pc is None or not category:
            return None
        k = f"{_clean_text(name)}||{_clean_text(category)}"
        idx = self.pc.key_to_idx.get(k)
        if idx is None:
            return None
        return F.normalize(self.pc.emb[idx].to(self.device), dim=-1)

    @torch.no_grad()
    def encode_label_for_sample(self, name: str, category: Optional[str], setting: str) -> torch.Tensor:
        setting = (setting or 'part_only').lower()
        vecs: List[torch.Tensor] = []
        if setting in ('part_plus_cat', 'ensemble'):
            v = self._lookup_pc(name, category)
            if v is not None:
                vecs.append(v)
        if setting in ('part_only', 'ensemble') or not category:
            v = self._lookup_po(name)
            if v is not None:
                vecs.append(v)
        if vecs:
            return F.normalize(torch.stack(vecs, dim=0).mean(dim=0), dim=-1)
        if self.fallback is not None:
            return self.fallback.encode_label_for_sample(name, category, setting)
        if self.strict:
            raise KeyError(f"Label not found in bank: name={name} cat={category} setting={setting}")
        return torch.zeros(self.text_dim, device=self.device)

    def encode_labels_for_sample(self, names: List[str], category: Optional[str], setting: str) -> torch.Tensor:
        if not names:
            return torch.empty(0, self.text_dim, device=self.device)
        vecs = [self.encode_label_for_sample(n, category, setting) for n in names]
        return torch.stack(vecs, dim=0)


def encode_text_from_part_names(seg_classes: Dict[str, List[int]], id2cat: Dict[int, str], *, device: torch.device,
                                setting: str, cache: BankTextCache | LRUTextCache) -> torch.Tensor:
    from train_patch_clip_parts_zs_point import PART_NAME_CANDIDATES as NAME_CANDS  # type: ignore
    max_gid = max((gid for gids in seg_classes.values() for gid in gids), default=-1)
    D = cache.text_dim
    bank = torch.zeros(max(50, max_gid + 1), D, device=device)
    filled = torch.zeros(bank.shape[0], dtype=torch.bool, device=device)
    for cat in id2cat.values():
        if cat not in seg_classes:
            continue
        gids = list(sorted(seg_classes[cat]))
        cand = NAME_CANDS.get(cat, [])
        part_names = [cand[i] if i < len(cand) else f'part{i}' for i in range(len(gids))]
        for i, gid in enumerate(gids):
            v = cache.encode_label_for_sample(part_names[i], cat if setting != 'part_only' else None, setting)
            v = F.normalize(v.to(device).float(), dim=-1)
            bank[gid] = v
            filled[gid] = True
    if (~filled).any():
        mean_vec = F.normalize(bank[filled].mean(dim=0, keepdim=True), dim=-1) if filled.any() else F.normalize(torch.randn(1, bank.shape[1], device=device), dim=-1)
        bank = torch.where(filled.unsqueeze(-1), bank, mean_vec.expand_as(bank))
    return bank


# -----------------------------
# Datasets

class NPZFolderDataset(Dataset):
    def __init__(self, roots: List[str], *, filename_contains: str = ''):
        self.files: List[Path] = []
        for r in roots:
            # Support quoted globs (e.g., ".../*_pts2048.npz") by expanding here
            if any(ch in r for ch in ['*', '?', '[']):
                self.files += [Path(p) for p in _glob.glob(r)]
                continue
            p = Path(r)
            if p.is_file() and p.suffix == '.npz':
                self.files.append(p)
            elif p.is_dir():
                self.files += list(p.rglob('*.npz'))
        if filename_contains:
            sub = str(filename_contains)
            self.files = [f for f in self.files if sub in f.name]
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under: {roots}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p, allow_pickle=True)
        pts = d['points'].astype(np.float32)
        labels = d['labels'].astype(np.int64)
        names = d['label_names'].tolist() if 'label_names' in d.files else []
        cat = (d['category'].item() if isinstance(d['category'], np.ndarray) else d['category']) if 'category' in d.files else ''
        slug = (d['slug'].item() if isinstance(d['slug'], np.ndarray) else d['slug']) if 'slug' in d.files else p.stem
        return {'points': pts, 'labels': labels, 'label_names': names, 'category': cat, 'slug': slug}


def collate_npz(batch):
    pts = np.stack([b['points'] for b in batch], axis=0)
    labs = np.stack([b['labels'] for b in batch], axis=0)
    return {
        'points': torch.from_numpy(pts),
        'labels': torch.from_numpy(labs),
        'label_names': [b.get('label_names', []) for b in batch],
        'category': [b.get('category', '') for b in batch],
        'slug': [b.get('slug', '') for b in batch],
    }


class ObjaverseGeneralDataset(Dataset):
    """Adapter for Objaverse-General-Find3D benchmark to the NPZ-like dict interface.
    Expects per-object folders with: points5000.pcd, labels.npy (0=unlabeled), label_map.json (id->name).
    Category is parsed from folder name prefix before the last underscore.
    """
    def __init__(self, root: str, *, split_json: str = '', use: str = 'all', npoints: int = 2048, names_json: str = '', debug: bool = False):
        self.root = Path(root)
        self.npoints = int(npoints)
        self.debug = bool(debug)
        if not self.root.exists():
            raise FileNotFoundError(f'Objaverse-General root not found: {self.root}')
        # Collect subfolders
        all_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        # Optional filter from split_json
        allow_cats: Optional[set[str]] = None
        if split_json:
            try:
                with open(split_json, 'r', encoding='utf-8') as f:
                    sj = json.load(f)
                if use == 'seen':
                    allow_cats = set(sj.get('seen_categories_sorted', []) or [])
                elif use == 'unseen':
                    allow_cats = set(sj.get('unseen_categories_sorted', []) or [])
            except Exception as e:
                if self.debug:
                    print(f"[objaverse] failed to read split_json={split_json}: {e}")
        # Optional override names per category
        self.cat2names: Dict[str, List[str]] = {}
        if names_json:
            try:
                with open(names_json, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.cat2names = {str(k).strip().lower(): list(v) for k, v in raw.items() if isinstance(v, (list, tuple))}
            except Exception as e:
                if self.debug:
                    print(f"[objaverse] failed to read names_json={names_json}: {e}")
        self.items: List[Path] = []
        for d in all_dirs:
            slug = d.name
            cat = slug.rsplit('_', 1)[0] if '_' in slug else slug
            if (allow_cats is not None) and (cat not in allow_cats):
                continue
            # Require expected files
            if not (d / 'labels.npy').exists():
                continue
            if not (d / 'label_map.json').exists():
                continue
            if not (d / 'points5000.pcd').exists():
                continue
            self.items.append(d)
        if not self.items:
            raise FileNotFoundError(f'No Objaverse-General objects found under {self.root}')

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _read_pcd_points(path: Path) -> np.ndarray:
        # Try Open3D first
        try:
            import open3d as o3d  # type: ignore
            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                raise RuntimeError('open3d returned invalid point cloud')
            return pts[:, :3].astype(np.float32)
        except Exception:
            # Minimal PCD reader (handles DATA ascii or DATA binary with float fields)
            with open(path, 'rb') as f:
                raw = f.read()
            header_end = None
            # Find end of header (line starting with DATA ... then newline)
            for term in [b'\nDATA binary\n', b'\r\nDATA binary\r\n', b'\nDATA ascii\n', b'\r\nDATA ascii\r\n', b'\nDATA binary_compressed\n']:
                pos = raw.find(term)
                if pos != -1:
                    header_end = pos + len(term)
                    data_mode = term.decode('latin-1').strip().split()[-1]
                    break
            if header_end is None:
                # Fallback: parse header lines to find DATA
                txt = raw.decode('latin-1', errors='ignore')
                lines = txt.splitlines()
                fields = []
                data_idx = None
                for i, ln in enumerate(lines):
                    if ln.startswith('FIELDS'):
                        fields = ln.split()[1:]
                    if ln.startswith('DATA'):
                        data_idx = i + 1
                        data_mode = ln.split()[1]
                        break
                if data_idx is None:
                    raise RuntimeError(f'PCD header parse failed for {path}')
                # ASCII mode
                if data_mode.lower().startswith('ascii'):
                    arr = np.loadtxt(lines[data_idx:])
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    idx_x = fields.index('x') if 'x' in fields else 0
                    idx_y = fields.index('y') if 'y' in fields else 1
                    idx_z = fields.index('z') if 'z' in fields else 2
                    return arr[:, [idx_x, idx_y, idx_z]].astype(np.float32)
                else:
                    raise RuntimeError(f'Unsupported PCD DATA mode for quick parse: {data_mode}')

            # Parse header (latin-1) for fields, sizes, counts, points
            header_txt = raw[:header_end].decode('latin-1', errors='ignore')
            fields = []
            sizes: List[int] = []
            counts: List[int] = []
            types: List[str] = []
            points = None
            for ln in header_txt.splitlines():
                if ln.startswith('FIELDS'):
                    fields = ln.split()[1:]
                elif ln.startswith('SIZE'):
                    sizes = [int(x) for x in ln.split()[1:]]
                elif ln.startswith('COUNT'):
                    counts = [int(x) for x in ln.split()[1:]]
                elif ln.startswith('TYPE'):
                    types = ln.split()[1:]
                elif ln.startswith('POINTS'):
                    try:
                        points = int(ln.split()[1])
                    except Exception:
                        points = None
            if (not fields) or (not sizes) or (not counts) or (points is None):
                raise RuntimeError(f'PCD header missing fields/sizes/counts/points for {path}')
            # Build dtype per field (assume little-endian)
            # Only float32 (F 4) and int/uint handled in a simple path
            # Flatten counts by repeating field names when COUNT>1
            expanded_fields: List[str] = []
            expanded_sizes: List[int] = []
            expanded_types: List[str] = []
            for name, sz, tp, cnt in zip(fields, sizes, types or ['F'] * len(fields), counts):
                for _ in range(int(cnt)):
                    expanded_fields.append(name)
                    expanded_sizes.append(int(sz))
                    expanded_types.append(tp)
            bpp = sum(expanded_sizes)
            data = raw[header_end:]
            # For common case of all float32
            if all((s == 4 and t.upper() == 'F') for s, t in zip(expanded_sizes, expanded_types)):
                need = points * (bpp // 4)
                arr = np.frombuffer(data[:need * 4], dtype='<f4', count=need)
                if arr.size != need:
                    raise RuntimeError(f'PCD binary size mismatch for {path}')
                arr = arr.reshape(points, len(expanded_fields))
            else:
                # Generic byte-structured view: build compound dtype
                dt_fields = []
                for i, (sz, tp) in enumerate(zip(expanded_sizes, expanded_types)):
                    if tp.upper() == 'F' and sz == 4:
                        dt = '<f4'
                    elif tp.upper() == 'F' and sz == 8:
                        dt = '<f8'
                    elif tp.upper() == 'I' and sz == 4:
                        dt = '<i4'
                    elif tp.upper() == 'U' and sz == 4:
                        dt = '<u4'
                    elif tp.upper() == 'U' and sz == 1:
                        dt = '|u1'
                    else:
                        # Fallback to raw bytes then cast to float
                        dt = f'|V{int(sz)}'
                    dt_fields.append(('f%d' % i, np.dtype(dt)))
                comp_dt = np.dtype(dt_fields)
                stride = comp_dt.itemsize
                if stride != bpp:
                    # Pad/align if needed
                    pass
                rec = np.frombuffer(data[:points * stride], dtype=comp_dt, count=points)
                # Build float array by casting numeric fields; unknown as 0
                arr = np.zeros((points, len(expanded_fields)), dtype=np.float32)
                for i in range(len(expanded_fields)):
                    try:
                        arr[:, i] = rec['f%d' % i].astype(np.float32)
                    except Exception:
                        pass
            # Extract x,y,z columns
            # If normals merged as separate fields, FIELDS contains normal_x, normal_y, normal_z
            try:
                ix = expanded_fields.index('x')
                iy = expanded_fields.index('y')
                iz = expanded_fields.index('z')
            except Exception:
                ix, iy, iz = 0, 1, 2
            pts = arr[:, [ix, iy, iz]].astype(np.float32)
            return pts

    def __getitem__(self, idx: int):
        d = self.items[idx]
        slug = d.name
        cat = slug.rsplit('_', 1)[0] if '_' in slug else slug
        # Points
        pts = self._read_pcd_points(d / 'points5000.pcd')
        # Labels: Objaverse uses 1..K (no unlabeled); keep raw here and normalize later
        labels = np.load(d / 'labels.npy').astype(np.int64)
        if labels.ndim > 1:
            labels = labels.reshape(-1)
        # Names: from label_map.json or override by category
        label_names: List[str] = []
        if self.cat2names.get(cat):
            label_names = list(self.cat2names[cat])
        else:
            try:
                with open(d / 'label_map.json', 'r', encoding='utf-8') as f:
                    lm = json.load(f)
                # Build contiguous list indexing from original id-1
                keyed = []
                for k, v in lm.items():
                    try:
                        kid = int(k)
                    except Exception:
                        continue
                    if kid <= 0:
                        continue
                    keyed.append((kid - 1, str(v)))
                if keyed:
                    max_id = max(i for i, _ in keyed)
                    names = [f'part {i}' for i in range(max_id + 1)]
                    for i, name in keyed:
                        if 0 <= i < len(names):
                            names[i] = name
                    label_names = names
            except Exception:
                label_names = []
        # Resample if requested (prefer precomputed FPS indices if available)
        if isinstance(self.npoints, int) and self.npoints > 0 and pts.shape[0] != self.npoints:
            # Prefer fully precomputed arrays if available
            pts_pre = d / f'points{self.npoints}_fps.npy'
            lbl_pre = d / f'labels{self.npoints}_fps.npy'
            if pts_pre.exists() and lbl_pre.exists():
                try:
                    pts2 = np.load(pts_pre)
                    lbl2 = np.load(lbl_pre)
                    if pts2.ndim == 2 and pts2.shape[1] >= 3 and pts2.shape[0] == self.npoints and lbl2.shape[0] == self.npoints:
                        pts = pts2[:, :3].astype(np.float32)
                        labels = lbl2.astype(np.int64)
                    else:
                        raise RuntimeError('precomputed shape mismatch')
                except Exception:
                    # fall back to idx/online
                    pass
            if pts.shape[0] != self.npoints:
                N = pts.shape[0]
                idx_path = d / f'idx_fps_{self.npoints}.npy'
                sel = None
                if idx_path.exists():
                    try:
                        sel = np.load(idx_path)
                        sel = sel.astype(np.int64)
                        if sel.ndim != 1:
                            sel = sel.reshape(-1)
                        if sel.size != self.npoints:
                            sel = None
                        elif sel.max(initial=-1) >= N or sel.min(initial=0) < 0:
                            sel = None
                    except Exception:
                        sel = None
                if sel is None:
                    if N >= self.npoints:
                        sel = np.random.choice(N, size=self.npoints, replace=False)
                    else:
                        sel = np.random.choice(N, size=self.npoints, replace=True)
                pts = pts[sel]
                labels = labels[sel]
        # Normalize label ids to 0..K-1 with -1 for unlabeled if present
        if labels.size > 0:
            # If labels are 1..K (common in Objaverse), shift to 0..K-1
            if labels.min(initial=0) >= 1:
                labels = labels - 1
            # If labels are 0..K (0 means unlabeled), map 0 -> -1 and shift to 0..K-1
            elif labels.min(initial=0) == 0:
                labels = labels - 1
        return {'points': pts.astype(np.float32), 'labels': labels.astype(np.int64), 'label_names': label_names, 'category': cat, 'slug': slug}


class ScanObjectNNPartsDataset(Dataset):
    """Adapter for ScanObjectNN parts format to the NPZ-like dict interface.
    Expects a directory containing category subfolders with .bin and *_part.bin
    files, and optionally a split_new.txt to filter train/test.
    """
    ID2CAT = {
        0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk', 6: 'display', 7: 'door',
        8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow', 12: 'sink', 13: 'sofa', 14: 'toilet'
    }
    CAT2ID = {v: k for k, v in ID2CAT.items()}
    PART_REMAP = {
        0: torch.arange(4), 1: torch.tensor([0, 1, 2, -1, 3]), 2: torch.arange(5), 3: torch.arange(7),
        4: torch.tensor([0, -1, 1, 2, 3, 4]), 5: torch.arange(4), 6: torch.arange(3), 7: torch.arange(3),
        8: torch.arange(4), 9: torch.arange(3), 10: torch.arange(3), 11: torch.arange(2), 12: torch.arange(4),
        13: torch.arange(5), 14: torch.arange(6)
    }

    def __init__(self, root: str, *, split: str = 'test', filter_background: bool = True, remap_labels: bool = True, npoints: int = 2048, names_json: str = '', debug: bool = False):
        self.root = Path(root)
        self.split = split
        self.filter_background = bool(filter_background)
        self.remap_labels = bool(remap_labels)
        self.npoints = int(npoints)
        self.debug = bool(debug)
        # Optional category->names mapping
        self.cat2names: Dict[str, List[str]] = {}
        if names_json:
            try:
                with open(names_json, 'r') as f:
                    data = json.load(f)
                    # normalize keys to lower-case
                    self.cat2names = {str(k).strip().lower(): list(v) for k, v in data.items() if isinstance(v, (list, tuple))}
            except Exception as e:
                if self.debug:
                    print(f"[scanobj] failed to load names_json={names_json}: {e}")
        # Resolve base folder: accept either direct dataset dir or suffixed variants
        candidates = [
            self.root,
            self.root / 'object_dataset_complete_with_parts',
            self.root / 'object_dataset_complete_with_parts_',
        ]
        base = None
        for c in candidates:
            if (c.exists() and c.is_dir() and any((c / d).is_dir() for d in self.ID2CAT.values())) or (c / 'split_new.txt').exists():
                base = c
                break
        self.base = base if base is not None else self.root
        if self.debug:
            print(f"[scanobj] root={repr(str(self.root))} exists={self.root.exists()} is_dir={self.root.is_dir()}")
            print(f"[scanobj] base={repr(str(self.base))} exists={self.base.exists()} is_dir={self.base.is_dir()}")
        # Try split file
        self.index: List[Tuple[str, str]] = []  # list of (category, stem)
        split_file = self.base / 'split_new.txt'
        if split_file.exists():
            try:
                with open(split_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()  # tab or space
                        if len(parts) < 3:
                            continue
                        file_rel, cat_id, is_train = parts[0], int(parts[1]), parts[2]
                        take = (self.split == 'train' and is_train != 't') or (self.split == 'test' and is_train == 't')
                        if not take:
                            continue
                        stem = file_rel[:-4] if file_rel.endswith('.txt') else Path(file_rel).stem
                        cat = self.ID2CAT.get(cat_id, Path(file_rel).parts[0])
                        self.index.append((cat, stem))
            except Exception:
                self.index = []
        if self.debug:
            print(f"[scanobj] split file used={split_file.exists()} collected={len(self.index)}")
        if not self.index:
            # Fallback: recursive scan for *_part.bin to be robust to unexpected layouts
            for pb in sorted(self.base.rglob('*_part.bin')):
                try:
                    cat = pb.parent.name.strip().lower()
                    stem = pb.name[:-9]  # strip _part.bin
                    # Ensure corresponding .bin exists
                    pc_bin = pb.parent / f'{stem}.bin'
                    if not pc_bin.exists():
                        continue
                    self.index.append((cat, stem))
                except Exception:
                    continue
        if self.debug:
            # Print a few discovered samples
            print(f"[scanobj] discovered pairs={len(self.index)} (showing up to 5)")
            for i, (cat, stem) in enumerate(self.index[:5]):
                print(f"  - {cat}/{stem}")
        if not self.index:
            raise FileNotFoundError(f'No ScanObjectNN parts found under {self.base}')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        cat, stem = self.index[idx]
        base = self.base / cat / stem
        pc_path = base.with_suffix('.bin')
        seg_path = base.parent / f'{stem}_part.bin'
        pts_f = np.fromfile(str(pc_path), dtype=np.float32)
        seg_f = np.fromfile(str(seg_path), dtype=np.float32)
        # Skip header float, then reshape
        if pts_f.size < 12 or seg_f.size < 3:
            raise RuntimeError(f'Corrupt bin files for {pc_path}')
        pts_arr = pts_f[1:].reshape(-1, 11)
        seg_arr = seg_f[1:].reshape(-1, 2)
        # Filter background by dominant instance label in last column of pts
        if self.filter_background:
            last = pts_arr[:, -1]
            mask = (last != 0) & (last != 1) & (last != 2)
            idxs = np.where(mask)[0]
            if idxs.size > 0:
                vals, cnts = np.unique(last[idxs], return_counts=True)
                keep_val = vals[np.argmax(cnts)]
                keep = (last == keep_val)
                pts_arr = pts_arr[keep]
                seg_arr = seg_arr[keep]
        pts = pts_arr[:, [0, 1, 2]].astype(np.float32)
        labels = seg_arr[:, -1].astype(np.int64)
        # Remap labels to contiguous ids per category; unknown -> -1
        if self.remap_labels:
            cid = self.CAT2ID.get(cat, None)
            if cid is not None:
                remap = self.PART_REMAP[cid].cpu().numpy()
                valid = (labels >= 0) & (labels < remap.shape[0])
                out = np.full(labels.shape[0], -1, dtype=np.int64)
                out[valid] = remap[labels[valid]]
                labels = out
        # Build label_names aligned to remapped ids if provided
        label_names: List[str] = []
        names_src = self.cat2names.get(cat, None)
        if names_src is not None and self.remap_labels and (cat in self.CAT2ID):
            cid = self.CAT2ID[cat]
            remap = self.PART_REMAP[cid].cpu().numpy()
            # New ids occupy set(remap[remap>=0]); map each new id to the corresponding source name index
            new_ids = sorted(set(int(x) for x in remap.tolist() if int(x) >= 0))
            if names_src:
                # construct array of size max(new_id)+1
                max_new = max(new_ids) if new_ids else -1
                label_names = [f'part {i}' for i in range(max_new + 1)]
                for raw_id, new_id in enumerate(remap.tolist()):
                    if new_id is None or int(new_id) < 0:
                        continue
                    rid = int(raw_id)
                    nid = int(new_id)
                    if rid < len(names_src):
                        label_names[nid] = str(names_src[rid])
        elif names_src is not None:
            # No remap applied; assume labels index directly into names_src
            label_names = list(names_src)
        # Resample to fixed npoints for batching
        N = pts.shape[0]
        if self.npoints > 0 and N != self.npoints:
            if N >= self.npoints:
                sel = np.random.choice(N, size=self.npoints, replace=False)
            else:
                sel = np.random.choice(N, size=self.npoints, replace=True)
            pts = pts[sel]
            labels = labels[sel]
        slug = f'{cat}/{stem}'
        return {'points': pts, 'labels': labels, 'label_names': label_names, 'category': cat, 'slug': slug}


# -----------------------------
# Geometry helpers and metrics

def compute_patch_targets_vector(point_labels: torch.Tensor, patch_idx: torch.Tensor, num_labels: int) -> torch.Tensor:
    G, M = patch_idx.shape
    gathered = point_labels.gather(0, patch_idx.reshape(-1)).view(G, M)
    lbl = (gathered + 1).clamp_min(0)
    K = int(num_labels) + 1
    one_hot = F.one_hot(lbl.clamp_max(K - 1), num_classes=K).sum(dim=1)
    one_hot[:, 0] = 0
    has_any = one_hot.sum(dim=1) > 0
    preds = one_hot.argmax(dim=1) - 1
    out = torch.full((G,), -1, dtype=torch.long, device=point_labels.device)
    out[has_any] = preds[has_any]
    return out


def assign_points_from_patches(points_xyz: torch.Tensor, patch_centers: torch.Tensor, patch_logits: torch.Tensor,
                               patch_idx: torch.Tensor, mode: str = 'nearest') -> torch.Tensor:
    B, _, N = points_xyz.shape
    K = patch_logits.shape[-1]
    if mode == 'membership':
        point_logits = torch.zeros(B, N, K, device=points_xyz.device)
        counts = torch.zeros(B, N, 1, device=points_xyz.device)
        for b in range(B):
            idx = patch_idx[b].reshape(-1)
            src = patch_logits[b].unsqueeze(1).expand_as(patch_idx[b].unsqueeze(-1)).reshape(-1, K)
            point_logits[b].index_add_(0, idx, src)
            ones = torch.ones(idx.shape[0], 1, device=points_xyz.device)
            counts[b].index_add_(0, idx, ones)
        return point_logits / counts.clamp_min(1.0)
    # nearest
    try:
        from knn_cuda import KNN  # type: ignore
        knn = KNN(k=1, transpose_mode=True)
        _, nearest = knn(patch_centers.transpose(1, 2).contiguous(), points_xyz.transpose(1, 2).contiguous())
        nearest = nearest.squeeze(-1)
    except Exception:
        dist = torch.cdist(points_xyz.transpose(1, 2), patch_centers.transpose(1, 2))
        nearest = dist.argmin(dim=-1)
    return patch_logits.gather(1, nearest.unsqueeze(-1).expand(-1, -1, K))


def compute_point_metrics(point_pred: torch.Tensor, target: torch.Tensor, label: torch.Tensor,
                          seg_classes: Dict[str, List[int]], id2cat: Dict[int, str]) -> Dict[str, float]:
    B, N = target.shape
    acc = (point_pred == target).float().mean().item()
    iou_list: List[float] = []
    cat_to_ious: Dict[str, List[float]] = {cat: [] for cat in seg_classes}
    for b in range(B):
        cat = id2cat[int(label[b].item())]
        valid = seg_classes[cat]
        preds = point_pred[b]
        gts = target[b]
        ious = []
        for pid in valid:
            pred_mask = preds == pid
            gt_mask = gts == pid
            inter = (pred_mask & gt_mask).sum().item()
            union = (pred_mask | gt_mask).sum().item()
            iou = 1.0 if union == 0 else inter / union
            ious.append(iou)
        if ious:
            inst = sum(ious) / len(ious)
            iou_list.append(inst)
            cat_to_ious[cat].append(inst)
    miou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    ciou_vals = [sum(v) / len(v) for v in cat_to_ious.values() if v]
    ciou = sum(ciou_vals) / len(ciou_vals) if ciou_vals else 0.0
    return dict(acc=acc, miou=miou, ciou=ciou)


# -----------------------------
# Dataset-specific evaluation

def build_text_bank_for_shapenet(*, device: torch.device, template_setting: str, text_backend: str,
                                 find3d_root: Optional[str]) -> Tuple[torch.Tensor, int, object, object]:
    """Returns (text_feats_50, text_dim, clip_model, tokenizer).
    Prefers offline text banks under <find3d_root>/labeled/text_banks.
    Falls back to open_clip online encoding if no bank is found.
    """
    # Attempt to load banks
    bank_dir = str(Path(find3d_root) / 'labeled' / 'text_banks') if find3d_root else None
    bank_po, bank_pc = try_load_text_banks(bank_dir)
    use_banks = (bank_po is not None) or (bank_pc is not None)
    clip_model = None
    tokenizer = None
    if use_banks:
        text_dim = bank_po.dim if bank_po is not None else bank_pc.dim
        cache = BankTextCache(device=device, bank_part_only=bank_po, bank_part_cat=bank_pc, fallback=None, strict=False)
        return torch.empty(0, text_dim, device=device), text_dim, clip_model, tokenizer  # placeholder; real bank built later with dataset seg_classes
    # Fallback: online CLIP encoder
    if open_clip is None:
        raise RuntimeError('open_clip is required when no offline text bank is available')
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    text_dim = int(clip_model.text_projection.shape[1])  # type: ignore[index]
    cache = LRUTextCache(device=device, backend='clip', capacity=20000, clip_model=clip_model, tokenizer=tokenizer, text_dim=text_dim)
    return torch.empty(0, text_dim, device=device), text_dim, clip_model, tokenizer


def evaluate_shapenet(ckpt_path: str, shapenet_root: str, *, device: torch.device, batch_size: int,
                      arch: str, num_group: int, group_size: int, template_setting: str, assign_mode: str,
                      find3d_root: Optional[str], text_backend: str = 'clip', progress: bool = True,
                      workers: int = 4, class_choice: Optional[List[str]] = None,
                      force_clip_text: bool = False) -> Dict[str, object]:
    # Build dataset/loader
    from data_utils.ShapeNetDataLoader import PartNormalDataset  # type: ignore
    sh_test = PartNormalDataset(root=shapenet_root, npoints=2048, split='test', normal_channel=False, color=False,
                                class_choice=class_choice)
    dl = DataLoader(sh_test, batch_size=batch_size, shuffle=False, num_workers=max(0, int(workers)))
    seg_classes = sh_test.seg_classes
    id2cat = {v: k for k, v in sh_test.classes.items()}

    # Build text backend (force CLIP if requested)
    clip_model = None; tokenizer = None
    if force_clip_text:
        if open_clip is None:
            raise RuntimeError('open_clip is required for --force_clip_text')
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', device=device)
        tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
        text_dim = int(clip_model.text_projection.shape[1])  # type: ignore[index]
        cache = LRUTextCache(device=device, backend='clip', capacity=20000, clip_model=clip_model, tokenizer=tokenizer, text_dim=text_dim)
    else:
        _, text_dim, clip_model, tokenizer = build_text_bank_for_shapenet(device=device, template_setting=template_setting,
                                                                          text_backend=text_backend, find3d_root=find3d_root)
        bank_dir = str(Path(find3d_root) / 'labeled' / 'text_banks') if find3d_root else None
        bank_po, bank_pc = try_load_text_banks(bank_dir)
        if (bank_po is not None) or (bank_pc is not None):
            cache = BankTextCache(device=device, bank_part_only=bank_po, bank_part_cat=bank_pc, fallback=None, strict=False)
        else:
            cache = LRUTextCache(device=device, backend='clip', capacity=20000, clip_model=clip_model, tokenizer=tokenizer, text_dim=text_dim)

    # Build model + projector for chosen text dim, then load ckpt
    model = build_model(arch=arch, num_group=num_group, group_size=group_size).to(device)
    proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)
    _ = load_ckpt(model, proj, ckpt_path)
    model.eval(); proj.eval()
    T50 = encode_text_from_part_names(seg_classes, id2cat, device=device, setting=template_setting, cache=cache)

    # Eval loop
    tot_patch = tot_patch_correct = 0
    pt_acc = pt_miou = pt_ciou = 0.0
    batches = 0
    cat_to_ious_global: Dict[str, List[float]] = {cat: [] for cat in seg_classes.keys()}
    with torch.no_grad():
        iterator = tqdm(dl, total=len(dl), desc='ShapeNetPart', smoothing=0.9) if progress else dl
        for points, label, target in iterator:
            points = points.to(device).float(); label = label.to(device).long(); target = target.to(device).long()
            # swap y/z then (B,C,N)
            points[:, :, [1, 2]] = points[:, :, [2, 1]]
            points = points.transpose(2, 1)
            xyz = points[:, :3, :]
            pe, pc, pi = model.forward_patches(points)  # type: ignore[operator]
            x = proj(pe)
            logits = (x @ T50.t()) / 0.07
            # category mask
            valid = torch.zeros_like(logits, dtype=torch.bool)
            for b in range(points.size(0)):
                cat = id2cat[int(label[b].item())]
                allow = seg_classes.get(cat, [])
                if allow:
                    valid[b, :, allow] = True
            logits = logits.masked_fill(~valid, -1e4)
            # patch acc
            for b in range(points.size(0)):
                lab_b = compute_patch_targets_vector(target[b], pi[b], 50)
                vl = lab_b >= 0
                if vl.any():
                    tot_patch += int(vl.sum().item())
                    patch_pred = logits[b].argmax(dim=-1)
                    tot_patch_correct += int((patch_pred[vl] == lab_b[vl]).sum().item())
            # point preds
            point_logits = assign_points_from_patches(xyz, pc, logits, pi, mode=assign_mode)
            point_pred = point_logits.argmax(dim=-1)
            metrics_batch = compute_point_metrics(point_pred, target, label, seg_classes, id2cat)
            pt_acc += metrics_batch['acc']; pt_miou += metrics_batch['miou']; pt_ciou += metrics_batch['ciou']
            # per-sample instance IoU by category
            for b in range(points.size(0)):
                cat = id2cat[int(label[b].item())]
                valid_ids = seg_classes.get(cat, [])
                ious = []
                for gid in valid_ids:
                    pm = (point_pred[b] == gid)
                    gm = (target[b] == gid)
                    inter = (pm & gm).sum().item()
                    union = (pm | gm).sum().item()
                    iou = 1.0 if union == 0 else (inter / union)
                    ious.append(iou)
                if ious:
                    inst_iou = float(sum(ious) / len(ious))
                    cat_to_ious_global.setdefault(cat, []).append(inst_iou)
            batches += 1

    patch_acc = tot_patch_correct / max(tot_patch, 1)
    point_acc = pt_acc / max(batches, 1)
    point_miou = pt_miou / max(batches, 1)
    point_ciou = pt_ciou / max(batches, 1)
    per_cat_iou = {cat: (sum(v) / len(v)) for cat, v in cat_to_ious_global.items() if v}
    return dict(patch_acc=patch_acc, point_acc=point_acc, point_miou=point_miou, point_ciou=point_ciou, per_cat_iou=per_cat_iou)


def _encode_texts_clip(names: List[str], *, category: Optional[str], setting: str, clip_model, tokenizer, device: torch.device) -> torch.Tensor:
    texts: List[str] = []
    for nm in names:
        nm = str(nm).replace('_', ' ').lower()
        if (setting == 'part_only') or (not category):
            texts += [f"{nm}", f"a {nm}", f"{nm} part"]
        else:
            cat = str(category).replace('_', ' ').lower()
            texts += [f"a {nm} of a {cat}", f"the {nm} of a {cat}", f"{cat} {nm}"]
    toks = tokenizer(texts).to(device)
    feat = F.normalize(clip_model.encode_text(toks), dim=-1)
    out = []
    i = 0
    for _ in names:
        v = F.normalize(feat[i:i + 3].mean(dim=0, keepdim=True), dim=-1)
        out.append(v.squeeze(0))
        i += 3
    return torch.stack(out, dim=0)


def _encode_texts_siglip(names: List[str], *, category: Optional[str], setting: str,
                         siglip_model, tokenizer, device: torch.device) -> torch.Tensor:
    # Mirrors _encode_texts_clip template behavior but with SigLIP encoder
    texts: List[str] = []
    for nm in names:
        nm = str(nm).replace('_', ' ').lower()
        if (setting == 'part_only') or (not category):
            texts += [f"{nm}", f"a {nm}", f"{nm} part"]
        else:
            cat = str(category).replace('_', ' ').lower()
            texts += [f"a {nm} of a {cat}", f"the {nm} of a {cat}", f"{cat} {nm}"]
    toks = tokenizer(texts, padding='max_length', return_tensors='pt')
    for k in toks:
        toks[k] = toks[k].to(device)
    with torch.no_grad():
        feat = siglip_model.get_text_features(**toks)
    feat = F.normalize(feat, dim=-1)
    out = []
    i = 0
    for _ in names:
        v = F.normalize(feat[i:i + 3].mean(dim=0, keepdim=True), dim=-1)
        out.append(v.squeeze(0))
        i += 3
    return torch.stack(out, dim=0)


def eval_npz_dataset(roots: List[str], *, ckpt_path: str, device: torch.device, batch_size: int, arch: str,
                     num_group: int, group_size: int, template_setting: str, assign_mode: str,
                     synonyms: Optional[Dict[str, List[str]]] = None, add_other_for_unlabeled: bool = False,
                     map_unlabeled_to_other: bool = False, unlabeled_as_category: bool = False,
                     unlabeled_category_use_templates: bool = False,
                     tau: float = 0.07, desc: str = 'NPZ', progress: bool = True, workers: int = 2,
                     npz_filter_substr: str = '', npz_names_json: str = '',
                     other_from_similarity: bool = False, other_sim_threshold: float = -1.0,
                     record_per_instance: bool = False, dump_path: str = '', save_predictions_dir: str = '') -> Dict[str, object]:
    ds = NPZFolderDataset(roots, filename_contains=npz_filter_substr)
    # Optional category -> list[str] part names override
    names_override: Optional[Dict[str, List[str]]] = None
    if npz_names_json:
        try:
            with open(npz_names_json, 'r') as f:
                raw = json.load(f)
            # normalize keys to lowercase for robust matching
            names_override = {str(k).strip().lower(): [str(x) for x in v] for k, v in raw.items() if isinstance(v, (list, tuple))}
        except Exception as e:
            print(f"[warn] failed to read npz_names_json={npz_names_json}: {e}")
    # Collate that can optionally resample each sample to a fixed size to allow batching
    def _collate_npz_resample(batch):
        target_npoints = int(os.getenv('NPZ_NPOINTS', '0'))
        # Fallback: if CLI passed --npz_npoints, we also attach it to env in main; but support env override here.
        pass
        # If no target set, try to auto-detect mismatch and force per-sample behavior
        if target_npoints <= 0:
            shapes = {b['points'].shape for b in batch}
            if len(shapes) == 1:
                return collate_npz(batch)
            # Variable sizes: default to per-sample stacking by resampling to the min size
            mins = min(s[0] for s in shapes)
            target_npoints = mins
        out = []
        for b in batch:
            pts = b['points']; labs = b['labels']
            N = int(pts.shape[0])
            if N != target_npoints:
                if N >= target_npoints:
                    idx = np.random.choice(N, size=target_npoints, replace=False)
                else:
                    idx = np.random.choice(N, size=target_npoints, replace=True)
                pts = pts[idx]; labs = labs[idx]
            out.append({'points': pts, 'labels': labs, 'label_names': b.get('label_names', []), 'category': b.get('category', ''), 'slug': b.get('slug', '')})
        return collate_npz(out)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=max(0, int(workers)), collate_fn=_collate_npz_resample)

    # Online CLIP for NPZ evaluation (banks are not used here typically)
    if open_clip is None:
        raise RuntimeError('open_clip required for NPZ evaluation')
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    text_dim = int(clip_model.text_projection.shape[1])  # type: ignore[index]

    model = build_model(arch=arch, num_group=num_group, group_size=group_size).to(device)
    proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)
    _ = load_ckpt(model, proj, ckpt_path)
    model.eval(); proj.eval(); clip_model.eval()

    @torch.no_grad()
    def _encode_names(names: List[str], category: Optional[str]) -> torch.Tensor:
        # Applies synonyms by averaging text embeddings if provided
        if not isinstance(names, (list, tuple)) or len(names) == 0:
            return _encode_texts_clip(['part'], category=category, setting=template_setting, clip_model=clip_model, tokenizer=tokenizer, device=device)
        groups = []
        for nm in names:
            vars_ = [str(nm)]
            if synonyms and nm in synonyms:
                s = synonyms[nm]
                if isinstance(s, str):
                    vars_.append(s)
                else:
                    vars_.extend(list(s))
            # unique
            seen = set(); vv = []
            for v in vars_:
                v2 = str(v).strip()
                if v2 and v2 not in seen:
                    vv.append(v2); seen.add(v2)
            groups.append(vv or [str(nm)])
        # flatten, encode, average per group
        flat = [v for g in groups for v in g]
        T_all = _encode_texts_clip(flat, category=category, setting=template_setting, clip_model=clip_model, tokenizer=tokenizer, device=device)
        out = []
        off = 0
        for g in groups:
            vec = F.normalize(T_all[off:off + len(g)].mean(dim=0, keepdim=True), dim=-1)
            out.append(vec.squeeze(0))
            off += len(g)
        return torch.stack(out, dim=0)

    tot_acc = 0.0; tot_miou = 0.0; batches = 0
    cat_to_ious_global: Dict[str, List[float]] = {}
    # Per-part IoU accumulators per category
    from collections import defaultdict
    part_iou_sum: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    part_iou_cnt: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    cat_to_names: Dict[str, List[str]] = {}
    # Optional JSONL per-instance dump
    dump_f = None
    global_index = 0
    if record_per_instance and dump_path:
        try:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            dump_f = open(dump_path, 'w')
        except Exception:
            dump_f = None

    # Optional directory to save per-instance predictions (NPZ)
    save_dir = str(save_predictions_dir or '').strip()
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass

    iterator = tqdm(dl, total=len(dl), desc=desc, smoothing=0.9) if progress else dl
    for batch in iterator:
        t_batch0 = time.perf_counter()
        pts_np = batch['points'].numpy()
        if pts_np.ndim == 2:
            pts_np = pts_np[None, ...]
        B = int(pts_np.shape[0])
        pts_list = []
        for i in range(B):
            p = pts_np[i]
            c = p.mean(axis=0, keepdims=True); p = p - c; r = (p ** 2).sum(axis=1) ** 0.5; m = r.max()
            if m > 0: p = p / m
            p = p.copy(); p[:, [1, 2]] = p[:, [2, 1]]
            pts_list.append(p)
        pts = torch.from_numpy(np.stack(pts_list, axis=0)).to(device).float().transpose(2, 1).contiguous()

        gt = batch['labels']
        if isinstance(gt, torch.Tensor):
            if gt.ndim == 1: gt = gt.unsqueeze(0)
        gt = gt.to(device).long()

        pe, pc, pi = model.forward_patches(pts)  # type: ignore[operator]
        feat = proj(pe)
        names_list = batch['label_names']
        cats_list = batch.get('category', [''] * B)
        if not isinstance(names_list, (list, tuple)) or len(names_list) != B:
            names_list = [names_list for _ in range(B)]
        if not isinstance(cats_list, (list, tuple)) or len(cats_list) != B:
            cats_list = [cats_list for _ in range(B)]

        acc_sum = 0.0; miou_sum = 0.0
        per_inst_records: List[dict] = []
        for i in range(B):
            names_i: Any = names_list[i]
            cat: str = str(cats_list[i])
            if not isinstance(names_i, (list, tuple)) or len(names_i) == 0:
                ul = torch.unique(gt[i]).detach().cpu().numpy().tolist()
                ul = [u for u in ul if u >= 0]
                names_i = [f'part {int(x)}' for x in ul] if ul else ['part 0']
            names: List[str] = [str(x) for x in names_i] if isinstance(names_i, (list, tuple)) else [str(names_i)]
            # Override with category-level names if provided
            if names_override is not None:
                cname = (cat or '').strip().lower()
                override = names_override.get(cname)
                if isinstance(override, list) and len(override) > 0:
                    # Ensure length covers present label ids; pad with generic names if needed
                    present = torch.unique(gt[i][gt[i] >= 0]).detach().cpu().numpy().tolist()
                    need = (max(present) + 1) if present else len(override)
                    if len(override) < need:
                        ov = list(override) + [f'part {j}' for j in range(len(override), need)]
                    else:
                        ov = override[:need]
                    names = ov
            # Remember names for this category (use the latest non-empty set; FAUST uses consistent global names)
            if isinstance(names, (list, tuple)) and len(names) > 0:
                cat_to_names[cat or ''] = list(names)
            # Encode base (non-'other') text features
            T_base = _encode_names(names, category=(cat or None)).to(device)
            T = T_base
            other_id = -1
            catlab_id = -1
            labeled_mask = (gt[i] >= 0)
            # If threshold-based 'other' is requested, ensure 'other' exists regardless of unlabeled GT
            force_add_other = bool(other_from_similarity) and (float(other_sim_threshold) >= 0.0)
            if (add_other_for_unlabeled and (~labeled_mask).any().item()) or force_add_other:
                other_names = ['other']
                T_other = _encode_texts_clip(other_names, category=(cat or None), setting='part_only', clip_model=clip_model, tokenizer=tokenizer, device=device)
                T_other = F.normalize(T_other.mean(dim=0, keepdim=True), dim=-1)
                T = torch.cat([T, T_other], dim=0)
                other_id = T.shape[0] - 1
            # Optional: add category label as a candidate class for unlabeled
            if unlabeled_as_category:
                cname = (cat or '').strip() or 'object'
                texts = [cname] if not unlabeled_category_use_templates else [cname, f'a {cname}', f'{cname} part']
                T_cat = _encode_texts_clip(texts, category=None, setting='part_only', clip_model=clip_model, tokenizer=tokenizer, device=device)
                T_cat = F.normalize(T_cat.mean(dim=0, keepdim=True), dim=-1)
                T = torch.cat([T, T_cat], dim=0)
                catlab_id = T.shape[0] - 1

            logits = (feat[i] @ T.t()) / max(tau, 1e-6)
            xyz_i = pts[i:i + 1, :3, :]
            pc_i = pc[i:i + 1]
            plog = assign_points_from_patches(xyz_i, pc_i, logits.unsqueeze(0), pi[i:i + 1], mode=assign_mode)
            pred = plog.argmax(dim=-1)[0]
            # Threshold-based 'other' assignment using cosine similarity (pre-temperature)
            if force_add_other and (other_id >= 0) and (T_base.shape[0] > 0):
                sims = (feat[i] @ T_base.t())  # (G, K_non_other)
                sims_pts = assign_points_from_patches(xyz_i, pc_i, sims.unsqueeze(0), pi[i:i + 1], mode=assign_mode)[0]  # (N, K_non_other)
                max_sim, _ = sims_pts.max(dim=-1)
                low_mask = max_sim < float(other_sim_threshold)
                if low_mask.any():
                    pred = pred.clone()
                    pred[low_mask] = int(other_id)
            # Accuracy on labeled points only (also store per-instance)
            inst_acc = 0.0
            if labeled_mask.any():
                inst_acc = float((pred[labeled_mask] == gt[i][labeled_mask]).float().mean().item())
                acc_sum += inst_acc

            if unlabeled_as_category and (catlab_id >= 0):
                # Map GT unlabeled to category label and include it in IoU
                gt_eff = gt[i].clone()
                gt_eff[~labeled_mask] = catlab_id
                present = torch.unique(gt_eff).detach().cpu().tolist()
                ious = []
                for c in present:
                    pm = (pred == int(c))
                    gm = (gt_eff == int(c))
                    inter = (pm & gm).sum().item(); union = (pm | gm).sum().item()
                    iou = 1.0 if union == 0 else (inter / union)
                    ious.append(iou)
                    kcat = cat or 'unknown'
                    part_iou_sum[kcat][int(c)] += float(iou)
                    part_iou_cnt[kcat][int(c)] += 1
                inst_iou = float(sum(ious) / len(ious)) if len(ious) > 0 else 0.0
                miou_sum += inst_iou
                cat_to_ious_global.setdefault(cat or 'unknown', []).append(inst_iou)
                # Optional: save per-instance NPZ (with effective GT)
                if save_dir:
                    try:
                        slug = batch.get('slug', [''])
                        sid = str(slug[i]) if isinstance(slug, (list, tuple)) and i < len(slug) else f"{cat}_{global_index + i}"
                        sub = os.path.join(save_dir, str(cat or ''))
                        os.makedirs(sub, exist_ok=True)
                        out_path = os.path.join(sub, f"{sid}.npz")
                        pts_raw = pts_np[i]
                        np.savez_compressed(out_path,
                                            points=pts_raw,
                                            pred=pred.detach().cpu().numpy(),
                                            gt=gt[i].detach().cpu().numpy(),
                                            gt_effective=gt_eff.detach().cpu().numpy(),
                                            cat_name=str(cat or ''),
                                            instance_id=sid,
                                            miou=float(inst_iou),
                                            label_names=np.array(names if isinstance(names, (list, tuple)) else [str(names)], dtype=object))
                    except Exception:
                        pass
                if dump_f is not None:
                    probs_pts = torch.softmax(plog, dim=-1)[0]
                    mean_max_prob = float(probs_pts.max(dim=-1).values.mean().item())
                    slug = batch.get('slug', [''])
                    per_inst_records.append({
                        'index': int(global_index + i),
                        'instance_id': str(slug[i] if isinstance(slug, (list, tuple)) and i < len(slug) else ''),
                        'slug': str(slug[i] if isinstance(slug, (list, tuple)) and i < len(slug) else ''),
                        'cat_name': str(cat or ''),
                        'cat_id': -1,
                        'inst_point_acc': float(inst_acc),
                        'miou': float(inst_iou),
                        'mean_max_prob': mean_max_prob,
                    })
            elif map_unlabeled_to_other and (other_id >= 0):
                # Map unlabeled GT points to 'other' and include 'other' in IoU
                gt_eff = gt[i].clone()
                gt_eff[~labeled_mask] = other_id
                present = torch.unique(gt_eff).detach().cpu().tolist()
                ious = []
                for c in present:
                    pm = (pred == int(c))
                    gm = (gt_eff == int(c))
                    inter = (pm & gm).sum().item(); union = (pm | gm).sum().item()
                    iou = 1.0 if union == 0 else (inter / union)
                    ious.append(iou)
                    # accumulate per-part IoU (including 'other')
                    kcat = cat or 'unknown'
                    part_iou_sum[kcat][int(c)] += float(iou)
                    part_iou_cnt[kcat][int(c)] += 1
                inst_iou = float(sum(ious) / len(ious)) if len(ious) > 0 else 0.0
                miou_sum += inst_iou
                cat_to_ious_global.setdefault(cat or 'unknown', []).append(inst_iou)
                if save_dir:
                    try:
                        slug = batch.get('slug', [''])
                        sid = str(slug[i]) if isinstance(slug, (list, tuple)) and i < len(slug) else f"{cat}_{global_index + i}"
                        sub = os.path.join(save_dir, str(cat or ''))
                        os.makedirs(sub, exist_ok=True)
                        out_path = os.path.join(sub, f"{sid}.npz")
                        pts_raw = pts_np[i]
                        np.savez_compressed(out_path,
                                            points=pts_raw,
                                            pred=pred.detach().cpu().numpy(),
                                            gt=gt_eff.detach().cpu().numpy(),
                                            cat_name=str(cat or ''),
                                            instance_id=sid,
                                            miou=float(inst_iou),
                                            label_names=np.array(names if isinstance(names, (list, tuple)) else [str(names)], dtype=object))
                    except Exception:
                        pass
                if dump_f is not None:
                    probs_pts = torch.softmax(plog, dim=-1)[0]
                    mean_max_prob = float(probs_pts.max(dim=-1).values.mean().item())
                    slug = batch.get('slug', [''])
                    per_inst_records.append({
                        'index': int(global_index + i),
                        'instance_id': str(slug[i] if isinstance(slug, (list, tuple)) and i < len(slug) else ''),
                        'slug': str(slug[i] if isinstance(slug, (list, tuple)) and i < len(slug) else ''),
                        'cat_name': str(cat or ''),
                        'cat_id': -1,
                        'inst_point_acc': float(inst_acc),
                        'miou': float(inst_iou),
                        'mean_max_prob': mean_max_prob,
                    })
            else:
                # Original behavior: IoU over present labeled GT only; exclude 'other'
                present = torch.unique(gt[i][labeled_mask]).detach().cpu().tolist() if labeled_mask.any() else []
                ious = []
                for c in present:
                    if other_id >= 0 and int(c) == int(other_id):
                        continue
                    pm = (pred == int(c)) & labeled_mask
                    gm = (gt[i] == int(c))
                    inter = (pm & gm).sum().item(); union = (pm | gm).sum().item()
                    iou = 1.0 if union == 0 else (inter / union)
                    ious.append(iou)
                    # accumulate per-part IoU
                    kcat = cat or 'unknown'
                    part_iou_sum[kcat][int(c)] += float(iou)
                    part_iou_cnt[kcat][int(c)] += 1
                inst_iou = float(sum(ious) / len(ious)) if len(ious) > 0 else 0.0
                miou_sum += inst_iou
                cat_to_ious_global.setdefault(cat or 'unknown', []).append(inst_iou)
                if save_dir:
                    try:
                        slug = batch.get('slug', [''])
                        sid = str(slug[i]) if isinstance(slug, (list, tuple)) and i < len(slug) else f"{cat}_{global_index + i}"
                        sub = os.path.join(save_dir, str(cat or ''))
                        os.makedirs(sub, exist_ok=True)
                        out_path = os.path.join(sub, f"{sid}.npz")
                        pts_raw = pts_np[i]
                        np.savez_compressed(out_path,
                                            points=pts_raw,
                                            pred=pred.detach().cpu().numpy(),
                                            gt=gt[i].detach().cpu().numpy(),
                                            cat_name=str(cat or ''),
                                            instance_id=sid,
                                            miou=float(inst_iou),
                                            label_names=np.array(names if isinstance(names, (list, tuple)) else [str(names)], dtype=object))
                    except Exception:
                        pass
                if dump_f is not None:
                    probs_pts = torch.softmax(plog, dim=-1)[0]
                    mean_max_prob = float(probs_pts.max(dim=-1).values.mean().item())
                    slug = batch.get('slug', [''])
                    per_inst_records.append({
                        'index': int(global_index + i),
                        'instance_id': str(slug[i] if isinstance(slug, (list, tuple)) and i < len(slug) else ''),
                        'slug': str(slug[i] if isinstance(slug, (list, tuple)) and i < len(slug) else ''),
                        'cat_name': str(cat or ''),
                        'cat_id': -1,
                        'inst_point_acc': float(inst_acc),
                        'miou': float(inst_iou),
                        'mean_max_prob': mean_max_prob,
                    })
        tot_acc += acc_sum / max(B, 1)
        tot_miou += miou_sum / max(B, 1)
        batches += 1
        if dump_f is not None and per_inst_records:
            t_batch1 = time.perf_counter()
            seconds = float(max(0.0, t_batch1 - t_batch0) / max(B, 1))
            for r in per_inst_records:
                r['seconds'] = seconds
                try:
                    dump_f.write(json.dumps(r) + '\n')
                except Exception:
                    pass
        global_index += B

    point_acc = tot_acc / max(batches, 1)
    point_miou = tot_miou / max(batches, 1)
    per_cat_iou = {cat: (sum(v) / len(v)) for cat, v in cat_to_ious_global.items() if v}
    point_ciou = (sum(per_cat_iou.values()) / len(per_cat_iou)) if per_cat_iou else 0.0
    # Build per-part IoU (average across samples), and attach names if available
    per_part_iou: Dict[str, Dict[int, float]] = {}
    per_part_names: Dict[str, List[str]] = {}
    for cat, dsum in part_iou_sum.items():
        diou = {}
        for pid, s in dsum.items():
            cnt = part_iou_cnt[cat].get(pid, 0)
            if cnt > 0:
                diou[int(pid)] = float(s / cnt)
        if diou:
            per_part_iou[cat] = diou
        if cat in cat_to_names:
            per_part_names[cat] = cat_to_names[cat]
    res = {'acc': point_acc, 'miou': point_miou, 'ciou': point_ciou, 'per_cat_iou': per_cat_iou, 'per_part_iou': per_part_iou, 'part_names': per_part_names}
    # Include info about 'other' similarity thresholding if used
    if other_from_similarity and (float(other_sim_threshold) >= 0.0):
        res.update({'other_from_similarity': True, 'other_sim_threshold': float(other_sim_threshold)})
    try:
        if dump_f is not None:
            dump_f.close()
    except Exception:
        pass
    return res


# -----------------------------
# CLI

def parse_args():
    p = argparse.ArgumentParser('Unified evaluator for ShapeNetPart, FAUST, PartNet-E, ScanObjectNN-Parts')
    p.add_argument('--dataset', type=str, action='append', choices=['shapenet', 'faust', 'partnete', 'scanobjectnn', 'objaverse'], required=True,
                   help='One or more datasets to evaluate; can repeat --dataset')
    p.add_argument('--ckpt', type=str, required=True, help='Path to trained checkpoint (.pt)')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--eval_seed', type=int, default=1234, help='Seed for deterministic eval; set workers=0 for strongest determinism')
    p.add_argument('--arch', type=str, default='pointtransformer', choices=['pointtransformer', 'ppat'])
    p.add_argument('--num_group', type=int, default=128)
    p.add_argument('--group_size', type=int, default=32)
    p.add_argument('--assign', type=str, default='nearest', choices=['nearest', 'membership'])
    p.add_argument('--text_setting', type=str, default='part_only', choices=['part_only', 'part_plus_cat', 'ensemble'])
    p.add_argument('--text_backend', type=str, default='clip', choices=['clip'])
    p.add_argument('--clip_model', type=str, default='ViT-bigG-14')
    p.add_argument('--clip_pretrained', type=str, default='laion2b_s39b_b160k')
    p.add_argument('--clip_tau', type=float, default=0.07)
    p.add_argument('--force_clip_text', action='store_true', default=False,
                   help='Ignore offline text bank and re-encode part names online with CLIP for ShapeNet eval')
    # Always use the training/evaluation script's evaluator for ShapeNetPart
    p.add_argument('--text_bank_require', action='store_true', default=False,
                   help='Require all labels to exist in bank (no fallback encodings)')
    p.add_argument('--drop_labels_not_in_bank', action='store_true', default=False,
                   help='If using a bank and not strict, drop labels not in the bank (avoid fallback)')
    p.add_argument('--eval_text_from_names', action='store_true', default=True,
                   help='Build eval text bank from human part names (like training)')
    p.add_argument('--debug', action='store_true', default=False, help='Print debug info about checkpoint/text dims and loads')
    p.add_argument('--no_progress', action='store_true', default=False, help='Disable tqdm progress bars')

    # ShapeNetPart
    p.add_argument('--shapenet_root', type=str, default='')
    p.add_argument('--shapenet_classes', type=str, default='',
                   help='CSV list of ShapeNet categories to include (e.g., "Chair,Table"). If empty, use all.')
    p.add_argument('--find3d_root', type=str, default='', help='Used to locate offline text banks if available')
    p.add_argument('--text_bank_dir', type=str, default='', help='Optional explicit text banks dir (overrides <find3d_root>/labeled/text_banks)')
    p.add_argument('--shapenet_names_json', type=str, default='',
                   help='Optional JSON mapping: category -> list of part names/prompts to build the ShapeNetPart text bank')

    # FAUST / PartNet-E
    p.add_argument('--faust_npz', type=str, action='append', default=[], help='Folder(s) or file(s) of FAUST NPZs')
    p.add_argument('--faust_synonyms_json', type=str, default='', help='JSON file mapping label->synonyms for FAUST')
    p.add_argument('--partnete_npz', type=str, action='append', default=[], help='Folder(s) or file(s) of PartNet-E NPZs')
    p.add_argument('--partnete_other', action='store_true', default=False, help="Add an 'other' label for unlabeled points")
    p.add_argument('--partnete_iou_map_unlabeled_to_other', action='store_true', default=False,
                   help="When computing IoU for PartNet-E, map GT unlabeled points to the added 'other' label and include it in IoU (requires --partnete_other)")
    p.add_argument('--partnete_iou_map_unlabeled_to_category', action='store_true', default=False,
                   help="For PartNet-E, also add the category name as a label and map GT unlabeled points to it; include it in IoU")
    p.add_argument('--partnete_category_use_templates', action='store_true', default=False,
                   help="When adding the category label, average simple templates ['cat', 'a cat', 'cat part'] instead of raw category only")
    p.add_argument('--partnete_other_from_similarity', action='store_true', default=False,
                   help="Enable assigning 'other' when the max non-'other' CLIP cosine similarity is below a threshold")
    p.add_argument('--partnete_other_sim_threshold', type=float, default=-1.0,
                   help="Cosine similarity threshold in [0,1] for assigning 'other' (requires --partnete_other_from_similarity)")
    p.add_argument('--npz_filter_substr', type=str, default='', help='Filter NPZ filenames by substring (e.g., "pts2048")')
    p.add_argument('--npz_names_json', type=str, default='', help='Optional JSON path: category -> list of part names (used instead of NPZ label_names)')

    # ScanObjectNN (parts)
    p.add_argument('--scanobj_root', type=str, default='', help='Root of ScanObjectNN parts (folder containing object_dataset_complete_with_parts*)')
    p.add_argument('--scanobj_split', type=str, default='test', choices=['train', 'test'])
    p.add_argument('--scanobj_filter_bg', action='store_true', default=True)
    p.add_argument('--scanobj_remap', action='store_true', default=True)
    p.add_argument('--scanobj_npoints', type=int, default=2048, help='Resample each ScanObjectNN cloud to this many points (with replacement if needed)')
    p.add_argument('--scanobj_names_json', type=str, default='', help='Optional JSON mapping category -> list of part names (index 0 is background)')

    # Objaverse-General benchmark
    p.add_argument('--objaverse_root', type=str, default='', help='Root of Objaverse-General benchmark (folder containing object subfolders)')
    p.add_argument('--objaverse_split_json', type=str, default='', help='Optional JSON from seen_unseen_objaverse_general.py to filter categories')
    p.add_argument('--objaverse_use', type=str, default='all', choices=['all', 'seen', 'unseen'], help='Select categories to include from split JSON')
    p.add_argument('--objaverse_npoints', type=int, default=2048, help='Resample each cloud to this many points (0=keep original, typically 5000)')
    p.add_argument('--objaverse_names_json', type=str, default='', help='Optional JSON: category -> list of part names to override label_map.json')
    p.add_argument('--objaverse_find3d', action='store_true', default=False,
                   help='Evaluate Objaverse-General in the Find3D style: add unlabeled column (0), IoU over 1..K only')
    p.add_argument('--siglip_model', type=str, default='google/siglip-base-patch16-224',
                   help='SigLIP text model to use when --objaverse_find3d is enabled')
    # Per-instance dump options
    p.add_argument('--dump_instances_shapenet', type=str, default='', help='Path to write per-instance JSONL records for ShapeNetPart (one JSON object per line)')
    p.add_argument('--save_instances_shapenet', type=str, default='', help='Directory to save per-instance predictions for ShapeNetPart (NPZ: points, pred, gt, meta)')
    p.add_argument('--dump_instances_faust', type=str, default='', help='Path to write per-instance JSONL records for FAUST (one JSON object per line)')
    p.add_argument('--save_instances_faust', type=str, default='', help='Directory to save per-instance predictions for FAUST (NPZ: points, pred, gt, meta)')
    # Single-instance cold-start timing
    p.add_argument('--single_instance_cold', action='store_true', default=False, help='Measure single-instance time including per-instance text feature generation (forces CLIP text for that measurement)')
    p.add_argument('--single_instance_index', type=int, default=0, help='Index of the ShapeNetPart test instance for single-instance timing')
    return p.parse_args()


def main():
    args = parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Deterministic behavior
    try:
        torch.manual_seed(args.eval_seed)
        np.random.seed(args.eval_seed)
        torch.cuda.manual_seed_all(args.eval_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    results: Dict[str, object] = {}
    ckpt_path = str(args.ckpt)

    if 'shapenet' in args.dataset:
        if not args.shapenet_root:
            raise ValueError('--shapenet_root is required when evaluating ShapeNetPart')
        # Import and use the same evaluator as training (evaluation script)
        from train_find3d_patch_clip_chairs_all_evaluation import (
            evaluate_shapenet_all, try_load_text_banks, BankTextCache, LRUTextCache as TrainLRUTextCache,
            encode_text_from_part_names,
        )
        # Dataset and loaders
        from data_utils.ShapeNetDataLoader import PartNormalDataset  # type: ignore
        class_choice = [s.strip() for s in args.shapenet_classes.split(',') if s.strip()] or None
        sh_test = PartNormalDataset(root=args.shapenet_root, npoints=2048, split='test', normal_channel=False, color=False,
                                    class_choice=class_choice)
        dl = DataLoader(sh_test, batch_size=args.batch_size, shuffle=False, num_workers=max(0, int(args.workers)))
        seg_classes = sh_test.seg_classes
        id2cat = {v: k for k, v in sh_test.classes.items()}
        cat_names = list(seg_classes.keys())

        # Text dim/banks identical path
        bank_dir = args.text_bank_dir or (str(Path(args.find3d_root) / 'labeled' / 'text_banks') if args.find3d_root else None)
        bank_po, bank_pc = try_load_text_banks(bank_dir)
        use_banks = (bank_po is not None) or (bank_pc is not None)
        if args.force_clip_text:
            use_banks = False
        clip_model = None; tokenizer = None
        if use_banks:
            text_dim = bank_po.dim if bank_po is not None else bank_pc.dim
        else:
            if open_clip is None:
                raise RuntimeError('open_clip not available; install open-clip-torch')
            clip_model, _, _ = open_clip.create_model_and_transforms(str(args.clip_model), pretrained=str(args.clip_pretrained), device=device)
            tokenizer = open_clip.get_tokenizer(str(args.clip_model))
            text_dim = int(clip_model.text_projection.shape[1])  # type: ignore[index]

        # Build model + projector, load checkpoint
        model = build_model(arch=args.arch, num_group=args.num_group, group_size=args.group_size).to(device)
        proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)
        _, load_logs = load_ckpt(model, proj, ckpt_path, verbose=args.debug)
        model.eval(); proj.eval()

        # Build cache and text bank exactly like training (with fallback when not strict)
        if use_banks:
            strict_mode = bool(args.text_bank_require and (not args.drop_labels_not_in_bank))
            fallback = None
            if not strict_mode:
                # Configure fallback to match bank backend/model to guarantee consistent dimensions
                bmeta = bank_po.meta if bank_po is not None else bank_pc.meta
                b_backend = str(bmeta.get('backend', 'clip'))
                if b_backend == 'clip':
                    b_clip_model = str(bmeta.get('clip_model', args.clip_model))
                    b_clip_pretrained = str(bmeta.get('clip_pretrained', args.clip_pretrained))
                    if open_clip is None:
                        raise RuntimeError('open_clip not available; install open-clip-torch')
                    fb_clip_model, _, _ = open_clip.create_model_and_transforms(b_clip_model, pretrained=b_clip_pretrained, device=device)
                    fb_tokenizer = open_clip.get_tokenizer(b_clip_model)
                    fallback = TrainLRUTextCache(backend='clip', device=device, capacity=5000,
                                                 clip_model=fb_clip_model, tokenizer=fb_tokenizer, hf_processor=None, hf_model=None, text_dim=text_dim)
                else:
                    # HF SigLIP fallback path
                    try:
                        from transformers import AutoProcessor, SiglipModel
                    except Exception as e:
                        raise RuntimeError('Please install transformers for SigLIP fallback') from e
                    b_siglip = str(bmeta.get('siglip_model', 'google/siglip-base-patch16-224'))
                    hf_processor = AutoProcessor.from_pretrained(b_siglip)
                    hf_text_model = SiglipModel.from_pretrained(b_siglip).to(device)  # type: ignore[attr-defined]
                    hf_text_model.eval()
                    fallback = TrainLRUTextCache(backend='siglip_hf', device=device, capacity=5000,
                                                 clip_model=None, tokenizer=None, hf_processor=hf_processor, hf_model=hf_text_model, text_dim=text_dim)
            cache = BankTextCache(device=device, bank_part_only=bank_po, bank_part_cat=bank_pc, fallback=fallback, strict=strict_mode)
        else:
            cache = LRUTextCache(device=device, backend='clip', capacity=20000, clip_model=clip_model, tokenizer=tokenizer, text_dim=text_dim)
        # Measure time to build the text features/bank used for evaluation
        tb0 = time.perf_counter()
        # Optional: override per-category part names from JSON (category -> list[str])
        names_override: Optional[Dict[str, List[str]]] = None
        if args.shapenet_names_json:
            try:
                with open(args.shapenet_names_json, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                names_override = {str(k).strip().lower(): [str(x) for x in v] for k, v in raw.items() if isinstance(v, (list, tuple))}
            except Exception as e:
                print(f"[warn] failed to read shapenet_names_json={args.shapenet_names_json}: {e}")

        if names_override is not None:
            text_feats_50 = torch.zeros(50, text_dim, device=device)
            for cat in cat_names:
                gids = sorted(seg_classes.get(cat, []))
                if not gids:
                    continue
                key = (cat or '').strip().lower()
                lst = names_override.get(key, [])
                if len(lst) < len(gids):
                    lst = list(lst) + [f'part{i}' for i in range(len(lst), len(gids))]
                for idx, gid in enumerate(gids):
                    nm = lst[idx]
                    v = cache.encode_label_for_sample(nm, (cat if args.text_setting != 'part_only' else None), args.text_setting)
                    text_feats_50[int(gid)] = F.normalize(v.to(device).float(), dim=-1)
        else:
            if args.eval_text_from_names:
                # The training/eval helper expects a list of category names here
                text_feats_50 = encode_text_from_part_names(seg_classes, cat_names, device=device, setting=args.text_setting, cache=cache)
            else:
                # Fallback generic texts if needed (clip backend)
                from train_find3d_patch_clip_chairs_all_evaluation import encode_shapenet_text as TrainEncodeSN
                text_feats_50 = TrainEncodeSN(seg_classes, cat_names, device=device, setting=args.text_setting,
                                              backend='clip', clip_model=clip_model, tokenizer=tokenizer, hf_processor=None, hf_model=None)
        tb1 = time.perf_counter()
        text_build_seconds = max(0.0, tb1 - tb0)

        # Optional: single-instance cold-start timing (build text only for that instance's category)
        single_text_seconds = None
        single_infer_seconds = None
        si_cat = None
        if args.single_instance_cold:
            if open_clip is None:
                raise RuntimeError('open_clip not available; install open-clip-torch to run --single_instance_cold')
            si_clip_model, _, _ = open_clip.create_model_and_transforms(str(args.clip_model), pretrained=str(args.clip_pretrained), device=device)
            si_tokenizer = open_clip.get_tokenizer(str(args.clip_model))
            si_cache = LRUTextCache(device=device, backend='clip', capacity=1024, clip_model=si_clip_model, tokenizer=si_tokenizer, text_dim=text_dim)
            # Prepare single-instance loader (exact sample by index)
            si_idx = max(0, int(args.single_instance_index))
            si_ds = Subset(sh_test, [si_idx])
            si_loader = DataLoader(si_ds, batch_size=1, shuffle=False, num_workers=0)
            # Determine instance category name
            pts0, cls0, seg0 = sh_test[si_idx]
            try:
                cls_val = int(cls0) if isinstance(cls0, (int, np.integer)) else int(np.array(cls0).reshape(-1)[0])
            except Exception:
                cls_val = int(np.array(cls0).reshape(-1)[0])
            si_cat = id2cat[int(cls_val)]
            # Build sparse 50xD bank for this category only
            allow = sorted(seg_classes.get(si_cat, []))
            bank_single = torch.zeros(50, text_dim, device=device)
            try:
                from train_patch_clip_parts_zs_point import PART_NAME_CANDIDATES as _NAME_CANDS  # type: ignore
            except Exception:
                _NAME_CANDS = {}
            cand = _NAME_CANDS.get(si_cat, [])
            part_names = [cand[i] if i < len(cand) else f'part{i}' for i in range(len(allow))]
            tb0_si = time.perf_counter()
            for i, gid in enumerate(allow):
                v = si_cache.encode_label_for_sample(part_names[i], si_cat if args.text_setting != 'part_only' else None, args.text_setting)
                bank_single[gid] = F.normalize(v.to(device).float(), dim=-1)
            tb1_si = time.perf_counter()
            single_text_seconds = max(0.0, tb1_si - tb0_si)
            # Inference on the single instance
            ti0 = time.perf_counter()
            _metrics_si, _ = evaluate_shapenet_all(
                model, proj, si_loader, seg_classes, id2cat, bank_single, args.clip_tau, device, args.assign, sample_cap=0,
                record_per_instance=False, dump_path=''
            )
            ti1 = time.perf_counter()
            single_infer_seconds = max(0.0, ti1 - ti0)

        # Measure full-test inference wall time and report normalized compute per 1000 instances
        test_size = len(sh_test)
        t0 = time.perf_counter()
        metrics, _samples = evaluate_shapenet_all(
            model, proj, dl, seg_classes, id2cat, text_feats_50, args.clip_tau, device, args.assign, sample_cap=0,
            record_per_instance=bool(args.dump_instances_shapenet), dump_path=str(args.dump_instances_shapenet or ''),
            save_predictions_dir=str(args.save_instances_shapenet or '')
        )
        t1 = time.perf_counter()
        wall_s = max(0.0, t1 - t0)
        sec_per_inst = (wall_s / test_size) if test_size > 0 else float('nan')
        sec_per_1000 = sec_per_inst * 1000.0 if test_size > 0 else float('nan')
        gpu_hours_per_1000 = sec_per_1000 / 3600.0 if test_size > 0 else float('nan')
        total_seconds_incl_text = wall_s + text_build_seconds
        sec_per_inst_incl_text = (total_seconds_incl_text / test_size) if test_size > 0 else float('nan')
        sec_per_1000_incl_text = sec_per_inst_incl_text * 1000.0 if test_size > 0 else float('nan')
        gpu_hours_per_1000_incl_text = sec_per_1000_incl_text / 3600.0 if test_size > 0 else float('nan')

        res = dict(metrics)
        res.update({
            'test_size': int(test_size),
            'inference_seconds': float(wall_s),
            'text_build_seconds': float(text_build_seconds),
            'sec_per_1000': float(sec_per_1000),
            'gpu_hours_per_1000': float(gpu_hours_per_1000),
            'inference_seconds_incl_text': float(total_seconds_incl_text),
            'sec_per_1000_incl_text': float(sec_per_1000_incl_text),
            'gpu_hours_per_1000_incl_text': float(gpu_hours_per_1000_incl_text),
        })
        if (single_text_seconds is not None) and (single_infer_seconds is not None):
            res.update({  # type: ignore[call-overload]
                'single_instance_index': int(max(0, int(args.single_instance_index))),
                'single_instance_category': str(si_cat),
                'single_instance_text_seconds': float(single_text_seconds),
                'single_instance_infer_seconds': float(single_infer_seconds),
                'single_instance_seconds_incl_text': float(single_text_seconds + single_infer_seconds),
            })
        if args.debug:
            print('[debug] metrics:', json.dumps(res, indent=2))
        results['shapenet'] = res

    synonyms = None
    if args.faust_synonyms_json:
        with open(args.faust_synonyms_json, 'r') as f:
            synonyms = json.load(f)

    if 'faust' in args.dataset:
        if not args.faust_npz:
            raise ValueError('--faust_npz required for FAUST evaluation')
        res = eval_npz_dataset(
            args.faust_npz, ckpt_path=ckpt_path, device=device, batch_size=args.batch_size, arch=args.arch,
            num_group=args.num_group, group_size=args.group_size, template_setting=args.text_setting,
            assign_mode=args.assign, synonyms=synonyms, add_other_for_unlabeled=False, map_unlabeled_to_other=False, tau=0.07,
            desc='FAUST', progress=(not args.no_progress), workers=args.workers, npz_filter_substr=str(args.npz_filter_substr or ''),
            npz_names_json=str(args.npz_names_json or ''),
            record_per_instance=bool(args.dump_instances_faust), dump_path=str(args.dump_instances_faust or ''),
            save_predictions_dir=str(args.save_instances_faust or ''),
        )
        results['faust'] = res

    if 'partnete' in args.dataset:
        if not args.partnete_npz:
            raise ValueError('--partnete_npz required for PartNet-E evaluation')
        res = eval_npz_dataset(
            args.partnete_npz, ckpt_path=ckpt_path, device=device, batch_size=args.batch_size, arch=args.arch,
            num_group=args.num_group, group_size=args.group_size, template_setting=args.text_setting,
            assign_mode=args.assign, synonyms=None, add_other_for_unlabeled=bool(args.partnete_other),
            map_unlabeled_to_other=bool(args.partnete_iou_map_unlabeled_to_other),
            unlabeled_as_category=bool(args.partnete_iou_map_unlabeled_to_category),
            unlabeled_category_use_templates=bool(args.partnete_category_use_templates), tau=0.07,
            desc='PartNet-E', progress=(not args.no_progress), workers=args.workers, npz_filter_substr=str(args.npz_filter_substr or ''),
            npz_names_json=str(args.npz_names_json or ''),
            other_from_similarity=bool(args.partnete_other_from_similarity),
            other_sim_threshold=float(args.partnete_other_sim_threshold),
        )
        results['partnete'] = res

    if 'scanobjectnn' in args.dataset:
        if not args.scanobj_root:
            raise ValueError('--scanobj_root required for ScanObjectNN parts evaluation')
        # Build dataset and reuse the same evaluation loop as NPZ by constructing the loader here
        ds = ScanObjectNNPartsDataset(args.scanobj_root, split=args.scanobj_split, filter_background=bool(args.scanobj_filter_bg), remap_labels=bool(args.scanobj_remap), npoints=int(args.scanobj_npoints), names_json=str(args.scanobj_names_json or ''), debug=bool(args.debug))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, int(args.workers)), collate_fn=collate_npz)

        if open_clip is None:
            raise RuntimeError('open_clip required for ScanObjectNN evaluation')
        clip_model, _, _ = open_clip.create_model_and_transforms(str(args.clip_model), pretrained=str(args.clip_pretrained), device=device)
        tokenizer = open_clip.get_tokenizer(str(args.clip_model))
        text_dim = int(clip_model.text_projection.shape[1])  # type: ignore[index]
        model = build_model(arch=args.arch, num_group=args.num_group, group_size=args.group_size).to(device)
        proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)
        _ = load_ckpt(model, proj, ckpt_path)
        model.eval(); proj.eval(); clip_model.eval()

        @torch.no_grad()
        def _encode_names_so(names: List[str], category: Optional[str]) -> torch.Tensor:
            if not isinstance(names, (list, tuple)) or len(names) == 0:
                return _encode_texts_clip(['part'], category=category, setting=args.text_setting, clip_model=clip_model, tokenizer=tokenizer, device=device)
            return _encode_texts_clip(names, category=category, setting=args.text_setting, clip_model=clip_model, tokenizer=tokenizer, device=device)

        tot_acc = 0.0; tot_miou = 0.0; batches = 0
        iterator = tqdm(dl, total=len(dl), desc='ScanObjectNN', smoothing=0.9) if not args.no_progress else dl
        tau = float(args.clip_tau)
        for batch in iterator:
            pts_np = batch['points'].numpy()
            if pts_np.ndim == 2:
                pts_np = pts_np[None, ...]
            B = int(pts_np.shape[0])
            pts_list = []
            for i in range(B):
                p = pts_np[i]
                c = p.mean(axis=0, keepdims=True); p = p - c; r = (p ** 2).sum(axis=1) ** 0.5; m = r.max()
                if m > 0: p = p / m
                p = p.copy(); p[:, [1, 2]] = p[:, [2, 1]]
                pts_list.append(p)
            pts = torch.from_numpy(np.stack(pts_list, axis=0)).to(device).float().transpose(2, 1).contiguous()

            gt = batch['labels']
            if isinstance(gt, torch.Tensor):
                if gt.ndim == 1: gt = gt.unsqueeze(0)
            gt = gt.to(device).long()

            pe, pc, pi = model.forward_patches(pts)  # type: ignore[operator]
            feat = proj(pe)
            names_list = batch['label_names']
            cats_list = batch.get('category', [''] * B)
            if not isinstance(names_list, (list, tuple)) or len(names_list) != B:
                names_list = [names_list for _ in range(B)]
            if not isinstance(cats_list, (list, tuple)) or len(cats_list) != B:
                cats_list = [cats_list for _ in range(B)]

            acc_sum = 0.0; miou_sum = 0.0
            for i in range(B):
                names_i_so: Any = names_list[i]
                cat_so: str = str(cats_list[i])
                if not isinstance(names_i_so, (list, tuple)) or len(names_i_so) == 0:
                    ul = torch.unique(gt[i]).detach().cpu().numpy().tolist()
                    ul = [u for u in ul if u >= 0]
                    names_i_so = [f'part {int(x)}' for x in ul] if ul else ['part 0']
                names_so: List[str] = [str(x) for x in names_i_so] if isinstance(names_i_so, (list, tuple)) else [str(names_i_so)]
                T = _encode_names_so(names_so, category=(cat_so or None)).to(device)
                labeled_mask = (gt[i] >= 0)
                logits = (feat[i] @ T.t()) / max(tau, 1e-6)
                xyz_i = pts[i:i + 1, :3, :]
                pc_i = pc[i:i + 1]
                plog = assign_points_from_patches(xyz_i, pc_i, logits.unsqueeze(0), pi[i:i + 1], mode=args.assign)
                pred = plog.argmax(dim=-1)[0]
                if labeled_mask.any():
                    acc_sum += (pred[labeled_mask] == gt[i][labeled_mask]).float().mean().item()
                present = torch.unique(gt[i][labeled_mask]).detach().cpu().tolist() if labeled_mask.any() else []
                ious = []
                for c in present:
                    pm = (pred == int(c)) & labeled_mask
                    gm = (gt[i] == int(c))
                    inter = (pm & gm).sum().item(); union = (pm | gm).sum().item()
                    iou = 1.0 if union == 0 else (inter / union)
                    ious.append(iou)
                if ious:
                    miou_sum += float(sum(ious) / len(ious))
            tot_acc += acc_sum / max(B, 1)
            tot_miou += miou_sum / max(B, 1)
            batches += 1

        results['scanobjectnn'] = {'acc': tot_acc / max(batches, 1), 'miou': tot_miou / max(batches, 1)}

    if 'objaverse' in args.dataset:
        if not args.objaverse_root:
            raise ValueError('--objaverse_root required for Objaverse-General evaluation')
        ds = ObjaverseGeneralDataset(args.objaverse_root, split_json=str(args.objaverse_split_json or ''), use=str(args.objaverse_use or 'all'), npoints=int(args.objaverse_npoints), names_json=str(args.objaverse_names_json or ''), debug=bool(args.debug))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, int(args.workers)), collate_fn=collate_npz)

        # Build model + projector
        use_find3d = bool(args.objaverse_find3d)
        if open_clip is None:
            raise RuntimeError('open_clip required for Objaverse-General evaluation')
        clip_model, _, _ = open_clip.create_model_and_transforms(str(args.clip_model), pretrained=str(args.clip_pretrained), device=device)
        tokenizer = open_clip.get_tokenizer(str(args.clip_model))
        text_dim = int(clip_model.text_projection.shape[1])  # type: ignore[index]

        model = build_model(arch=args.arch, num_group=args.num_group, group_size=args.group_size).to(device)
        proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)
        _ = load_ckpt(model, proj, ckpt_path)
        model.eval(); proj.eval()
        clip_model.eval()

        # Cache: (category, names tuple) -> text embeddings
        text_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor] = {}
        for i in range(len(ds)):
            try:
                rec = ds[i]
            except Exception:
                continue
            cat = (rec.get('category') or '').strip().lower()
            names = rec.get('label_names') or []
            key = (cat, tuple([str(x) for x in names]))
            if key in text_cache:
                continue
            names_list = list(key[1]) or ['part']
            T = _encode_texts_clip(names_list, category=(cat or None), setting=args.text_setting, clip_model=clip_model, tokenizer=tokenizer, device=device)
            text_cache[key] = T.detach().clone().cpu()

        @torch.no_grad()
        def _encode_names_obj(names: List[str], category: Optional[str]) -> torch.Tensor:
            nm = [str(x) for x in (names or [])]
            cat_k = (category or '').strip().lower()
            key = (cat_k, tuple(nm))
            if key in text_cache:
                return text_cache[key].to(device)
            T = _encode_texts_clip(nm if nm else ['part'], category=(category or None), setting=args.text_setting, clip_model=clip_model, tokenizer=tokenizer, device=device)
            text_cache[key] = T.detach().clone().cpu()
            return T

        # Iterate and evaluate
        iterator = tqdm(dl, total=len(dl), desc='Objaverse-General', smoothing=0.9) if not args.no_progress else dl
        tau = float(args.clip_tau)
        tot_acc = 0.0; tot_miou = 0.0; batches = 0
        for batch in iterator:
            pts_np = batch['points'].numpy()
            if pts_np.ndim == 2:
                pts_np = pts_np[None, ...]
            B = int(pts_np.shape[0])
            pts_list = []
            for i in range(B):
                p = pts_np[i]
                c = p.mean(axis=0, keepdims=True); p = p - c; r = (p ** 2).sum(axis=1) ** 0.5; m = r.max()
                if m > 0: p = p / m
                p = p.copy(); p[:, [1, 2]] = p[:, [2, 1]]
                pts_list.append(p)
            pts = torch.from_numpy(np.stack(pts_list, axis=0)).to(device).float().transpose(2, 1).contiguous()

            gt = batch['labels']
            if isinstance(gt, torch.Tensor) and gt.ndim == 1:
                gt = gt.unsqueeze(0)
            gt = gt.to(device).long()

            pe, pc, pi = model.forward_patches(pts)  # type: ignore[operator]
            feat = proj(pe)
            names_list = batch['label_names']
            cats_list = batch.get('category', [''] * B)
            if not isinstance(names_list, (list, tuple)) or len(names_list) != B:
                names_list = [names_list for _ in range(B)]
            if not isinstance(cats_list, (list, tuple)) or len(cats_list) != B:
                cats_list = [cats_list for _ in range(B)]

            acc_sum = 0.0; miou_sum = 0.0
            for i in range(B):
                names_i_obj: Any = names_list[i]
                cat_obj: str = str(cats_list[i])
                if not isinstance(names_i_obj, (list, tuple)) or len(names_i_obj) == 0:
                    ul = torch.unique(gt[i]).detach().cpu().numpy().tolist()
                    ul = [u for u in ul if u >= 0]
                    names_i_obj = [f'part {int(x)}' for x in ul] if ul else ['part 0']
                names_obj: List[str] = [str(x) for x in names_i_obj] if isinstance(names_i_obj, (list, tuple)) else [str(names_i_obj)]
                T = _encode_names_obj(names_obj, category=(cat_obj or None)).to(device)
                labeled_mask = (gt[i] >= 0)
                logits = (feat[i] @ T.t()) / max(tau, 1e-6)  # (G,K)
                if use_find3d:
                    # Append unlabeled column (exactly 0) like Find3D and argmax over 0..K
                    zeros_col = torch.zeros(logits.shape[0], 1, device=logits.device, dtype=logits.dtype)
                    logits = torch.cat([zeros_col, logits], dim=-1)  # (G, K+1)
                xyz_i = pts[i:i + 1, :3, :]
                pc_i = pc[i:i + 1]
                plog = assign_points_from_patches(xyz_i, pc_i, logits.unsqueeze(0), pi[i:i + 1], mode=args.assign)
                pred = plog.argmax(dim=-1)[0]
                # Accuracy on labeled points only
                if labeled_mask.any():
                    acc_sum += (pred[labeled_mask] == (gt[i][labeled_mask] + (1 if use_find3d else 0))).float().mean().item() if use_find3d else (pred[labeled_mask] == gt[i][labeled_mask]).float().mean().item()
                # IoU
                ious = []
                if use_find3d:
                    # Map GT to 1..K with 0 for unlabeled; evaluate over 1..K only
                    gt1k = gt[i].clone()
                    gt1k[gt1k >= 0] = gt1k[gt1k >= 0] + 1
                    gt1k[gt1k < 0] = 0
                    K = T.shape[0]
                    for c in range(1, K + 1):
                        pm = (pred == int(c))
                        gm = (gt1k == int(c))
                        inter = (pm & gm).sum().item(); union = (pm | gm).sum().item()
                        iou = 1.0 if union == 0 else (inter / union)
                        ious.append(iou)
                else:
                    present = torch.unique(gt[i][labeled_mask]).detach().cpu().tolist() if labeled_mask.any() else []
                    for c in present:
                        pm = (pred == int(c)) & labeled_mask
                        gm = (gt[i] == int(c))
                        inter = (pm & gm).sum().item(); union = (pm | gm).sum().item()
                        iou = 1.0 if union == 0 else (inter / union)
                        ious.append(iou)
                if ious:
                    miou_sum += float(sum(ious) / len(ious))
            tot_acc += acc_sum / max(B, 1)
            tot_miou += miou_sum / max(B, 1)
            batches += 1

        results['objaverse'] = {'acc': tot_acc / max(batches, 1), 'miou': tot_miou / max(batches, 1), 'find3d_style': use_find3d}

    # Pretty print JSON summary
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
