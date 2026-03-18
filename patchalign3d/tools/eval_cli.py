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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
            setting: Template setting — ``'part_only'``,
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
