"""
Self-contained PointTransformer encoder matching the training architecture used in
segmentation/models/PointTransformer_patched.get_model (defaults: trans_dim=384, depth=12, num_heads=6,
group_size=32, num_group=128, encoder_dims=256). This file exposes a drop-in encoder so other
applications can import it and load trained weights easily.

Key entry points:
- class PointTransformer: encoder-only module with forward_patches(...) and forward(...)
- function build_default_encoder(...): convenience factory with training defaults
- method load_weights(...): robust loader for checkpoints saved by our trainer
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from timm.layers.drop import DropPath

try:
    from knn_cuda import KNN  # CUDA fast KNN
except Exception:  # pragma: no cover
    KNN = None

try:
    from pointnet2_ops import pointnet2_utils
except Exception:  # pragma: no cover
    pointnet2_utils = None


def fps(xyz: torch.Tensor, number: int) -> torch.Tensor:
    """Furthest Point Sampling using pointnet2_ops if available.
    xyz: (B, N, 3) float; returns sampled points (B, number, 3)
    """
    if pointnet2_utils is None:
        # Fallback: naive stride as a last resort (not ideal, but avoids crash)
        B, N, _ = xyz.shape
        idx = torch.linspace(0, N - 1, steps=number, device=xyz.device).long()
        idx = idx.clamp_max(N - 1)
        gathered = torch.gather(xyz, 1, idx.view(1, -1, 1).expand(B, -1, 3))
        return gathered
    fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), number)
    fps_data = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # type: ignore[union-attr]
    return fps_data


class Group(nn.Module):
    """Patch grouping with FPS centers and KNN neighborhoods.
    - Accepts points with C>=3 (xyz | [extra]) and preserves extra channels.
    Returns (neighborhood, center, idx_rel):
      neighborhood: (B, G, M, C) with xyz normalized per-center
      center:       (B, G, 3)
      idx_rel:      (B, G, M) indices into each sample's N points
    """
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self._knn = KNN(k=self.group_size, transpose_mode=True) if KNN is not None else None

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = xyz.shape
        if C > 3:
            base = xyz[:, :, :3].contiguous()
            extra = xyz[:, :, 3:].contiguous()
        else:
            base = xyz.contiguous()
            extra = None

        center = fps(base, self.num_group)  # (B, G, 3)

        if self._knn is not None:
            _, idx = self._knn(base, center)  # (B, G, M)
        else:
            # CPU fallback using pairwise distance
            d = torch.cdist(center, base)  # (B,G,N)
            idx = d.topk(k=self.group_size, dim=-1, largest=False)[1]

        idx_rel = idx.clone()
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx_flat = (idx + idx_base).view(-1)

        neigh_xyz = base.view(B * N, -1)[idx_flat, :].view(B, self.num_group, self.group_size, 3).contiguous()
        if extra is not None:
            nc = extra.shape[-1]
            neigh_ex = extra.view(B * N, -1)[idx_flat, :].view(B, self.num_group, self.group_size, nc).contiguous()

        neigh_xyz = neigh_xyz - center.unsqueeze(2)
        neighborhood = torch.cat((neigh_xyz, neigh_ex), dim=-1) if extra is not None else neigh_xyz
        return neighborhood, center, idx_rel


class Encoder(nn.Module):
    """Local patch encoder from PointTransformer (with optional color/extra dims).
    Input: (B, G, M, C) where C=3 or 3+F (e.g., xyz(+rgb)).
    Output: (B, G, encoder_channel)
    """
    def __init__(self, encoder_channel: int, color: bool = False):
        super().__init__()
        in_ch = 6 if color else 3
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_ch, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        B, G, M, C = point_groups.shape
        x = point_groups.reshape(B * G, M, C).transpose(1, 2).contiguous()  # (BG, C, M)
        feat = self.first_conv(x)                                           # (BG, 256, M)
        feat_g = feat.max(dim=2, keepdim=True)[0]                           # (BG, 256, 1)
        feat = torch.cat([feat_g.expand(-1, -1, M), feat], dim=1)           # (BG, 512, M)
        feat = self.second_conv(feat)                                       # (BG, Cenc, M)
        feat_g = feat.max(dim=2, keepdim=False)[0]                          # (BG, Cenc)
        return feat_g.view(B, G, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder that returns features from selected layers, matching training.
    fetch_idx = [3, 7, 11] when depth=12.
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx and i < len(self.blocks):
                feature_list.append(x)
        # If depth < max(fetch_idx)+1, fall back to returning the last 3 feature maps
        if not feature_list:
            feature_list = [x]
        return feature_list


class PointTransformer(nn.Module):
    """Encoder-only PointTransformer with the same parameter names as training.
    Provides forward_patches(...) to obtain OUTPUT PATCH FEATURES.
    """
    def __init__(self,
                 trans_dim: int = 384,
                 depth: int = 12,
                 drop_path_rate: float = 0.1,
                 num_heads: int = 6,
                 group_size: int = 32,
                 num_group: int = 128,
                 encoder_dims: int = 256,
                 color: bool = True):
        super().__init__()
        # mirror training-time attribute names
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims
        self.color = color

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, color=self.color)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,  # type: ignore[arg-type]
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    @torch.no_grad()
    def load_weights(self, ckpt_path: str) -> Dict[str, int]:
        """Load weights from a trainer checkpoint or a raw state_dict.
        Accepts keys wrapped with 'module.' or 'point_encoder.' and ignores extras.
        Returns a dict with counts of loaded, missing, and unexpected keys.
        """
        obj = torch.load(ckpt_path, map_location='cpu')
        sd = obj
        if isinstance(obj, dict):
            if 'model' in obj and isinstance(obj['model'], dict):
                sd = obj['model']
            elif 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                sd = obj['state_dict']
        new_sd = {}
        for k, v in sd.items():
            k2 = k
            if k2.startswith('module.'):
                k2 = k2[len('module.'):]
            if k2.startswith('point_encoder.'):
                k2 = k2[len('point_encoder.'):]
            new_sd[k2] = v

        current = self.state_dict()
        loadable = {k: v for k, v in new_sd.items() if (k in current and current[k].shape == v.shape)}
        missing = [k for k in current.keys() if k not in loadable]
        unexpected = [k for k in new_sd.keys() if k not in current]
        self.load_state_dict(loadable, strict=False)
        return {'loaded': len(loadable), 'missing': len(missing), 'unexpected': len(unexpected)}

    def forward_patches(self, pts: torch.Tensor):
        """Encoder-only path returning output patch embeddings, centers and membership indices.
        pts: (B, C, N) with C>=3 (xyz | [extra])
        Returns: (patch_emb: B, trans_dim, G), (patch_centers: B, 3, G), (patch_indices: B, G, M)
        """
        _B, _C, _N = pts.shape
        pts_bn = pts.transpose(-1, -2).contiguous()  # (B, N, C)
        neighborhood, center, patch_idx = self.group_divider(pts_bn)  # (B,G,M,C?), (B,G,3), (B,G,M)
        group_input_tokens = self.encoder(neighborhood)               # (B, G, encoder_dims)
        group_input_tokens = self.reduce_dim(group_input_tokens)      # (B, G, trans_dim)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        feature_list_raw = self.blocks(x, pos)                        # list of (B, G+1, C)
        # normalize and drop cls, transpose to (B, C, G)
        feature_list = [self.norm(t)[:, 1:].transpose(-1, -2).contiguous() for t in feature_list_raw]
        patch_emb = feature_list[-1]
        patch_centers = center.transpose(-1, -2).contiguous()
        return patch_emb, patch_centers, patch_idx

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """Convenience forward returning only the final patch features (B, C, G)."""
        pe, _, _ = self.forward_patches(pts)
        return pe


def build_default_encoder(**overrides) -> PointTransformer:
    """Factory with training defaults, override any arg by name."""
    args = dict(trans_dim=384, depth=12, drop_path_rate=0.1, num_heads=6,
                group_size=32, num_group=128, encoder_dims=256, color=True)
    args.update(overrides)
    return PointTransformer(**args)  # type: ignore[arg-type]


__all__ = ['PointTransformer', 'build_default_encoder', 'Group', 'Encoder', 'TransformerEncoder']
