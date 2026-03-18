"""
Thin wrapper around your existing models/PointTransformer.py
- swaps the Group to also return patch membership indices
- adds get_model.forward_patches(...) that returns OUTPUT PATCH FEATURES
"""

import importlib
import torch
import torch.nn as nn
from knn_cuda import KNN

# Import the original PointTransformer module (must be in sys.path)
PT = importlib.import_module('PointTransformer')


class PatchedGroup(nn.Module):
    """
    Drop-in replacement for PointTransformer.Group that also returns KNN indices (per-patch membership).
    Returns:
      neighborhood : (B, G, M, C)   (xyz normalized per center + optional extra channels)
      center       : (B, G, 3)
      idx_rel      : (B, G, M)      indices into the per-sample N points (NO batch base added)
    """
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        xyz: (B, N, C)   where C>=3 (xyz | [extra-feats])
        """
        batch_size, num_points, C = xyz.shape
        if C > 3:
            data = xyz
            xyz_only = data[:, :, :3].contiguous()
            extra = data[:, :, 3:].contiguous()
        else:
            xyz_only = xyz.contiguous()
            extra = None

        # FPS centers on xyz
        center = PT.fps(xyz_only, self.num_group)  # (B, G, 3)

        # KNN to get neighborhoods
        _, idx = self.knn(xyz_only, center)  # (B, G, M)
        idx_rel = idx.clone()                # keep relative indices for labels

        # Gather neighborhoods (flattened indexing)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_flat = (idx + idx_base).view(-1)
        neigh_xyz = xyz_only.view(batch_size * num_points, -1)[idx_flat, :]
        neigh_xyz = neigh_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        if extra is not None:
            neigh_extra = extra.view(batch_size * num_points, -1)[idx_flat, :]
            neigh_extra = neigh_extra.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz by center
        neigh_xyz = neigh_xyz - center.unsqueeze(2)  # (B,G,M,3)

        if extra is not None:
            neighborhood = torch.cat((neigh_xyz, neigh_extra), dim=-1)  # (B,G,M,3+F)
        else:
            neighborhood = neigh_xyz

        return neighborhood, center, idx_rel


class BlockNoResidualFFN(nn.Module):
    """Wrap a transformer block but drop the residual connection and FFN."""
    def __init__(self, base_block):
        super().__init__()
        self.norm1 = base_block.norm1
        self.attn = base_block.attn

    def forward(self, x):
        return self.attn(self.norm1(x))


class get_model(PT.get_model):
    """
    Inherit your original get_model and:
    - replace self.group_divider so we also get indices
    - add forward_patches(...) to expose OUTPUT PATCH FEATURES + centers + membership
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # swap the grouper to the patched one (same API but returns idx too)
        self.group_divider = PatchedGroup(num_group=self.num_group, group_size=self.group_size)
        if getattr(config, 'last_block_no_residual', False):
            self._strip_last_block_residual()

    def _strip_last_block_residual(self):
        encoder = getattr(self, 'blocks', None)
        if encoder is None or not hasattr(encoder, 'blocks'):
            return
        if len(encoder.blocks) == 0:
            return
        encoder.blocks[-1] = BlockNoResidualFFN(encoder.blocks[-1])

    def forward_patches(self, pts):
        """
        Encoder-only path that returns patch embeddings (OUTPUT PATCH FEATURES), centers, and membership indices.

        Args:
            pts: (B, C, N)   C>=3  (xyz | [extra])
        Returns:
            patch_emb    : (B, trans_dim, G)  deepest transformer tokens (your output patch features)
            patch_centers: (B, 3, G)
            patch_indices: (B, G, M)         indices into each sample's N points
        """
        _B, _C, _N = pts.shape
        pts_bn = pts.transpose(-1, -2).contiguous()  # (B,N,C) for grouping

        # Group neighborhoods; now we also get indices
        neighborhood, center, patch_idx = self.group_divider(pts_bn)  # (B,G,M,C?), (B,G,3), (B,G,M)

        # Encode neighborhoods to tokens
        group_input_tokens = self.encoder(neighborhood)           # (B, G, encoder_dims)
        group_input_tokens = self.reduce_dim(group_input_tokens)  # (B, G, trans_dim)

        # Build transformer input with CLS token + positional embedding from centers
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos    = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos        = self.pos_embed(center)                       # (B, G, trans_dim)

        x   = torch.cat((cls_tokens, group_input_tokens), dim=1)  # (B, G+1, trans_dim)
        pos = torch.cat((cls_pos, pos), dim=1)

        # Feature list from requested layers; same as your normal forward path
        feature_list_raw = self.blocks(x, pos)  # list of (B, G+1, trans_dim)
        feature_list = [self.norm(t)[:, 1:].transpose(-1, -2).contiguous() for t in feature_list_raw]  # each (B,C,G)

        patch_emb     = feature_list[-1]                           # deepest = output patch features
        patch_centers = center.transpose(-1, -2).contiguous()      # (B,3,G)
        return patch_emb, patch_centers, patch_idx

    def forward_patches_with_cls(self, pts):
        """Same as forward_patches but also returns final CLS token (B, trans_dim)."""
        _B, _C, _N = pts.shape
        pts_bn = pts.transpose(-1, -2).contiguous()  # (B,N,C) for grouping

        neighborhood, center, patch_idx = self.group_divider(pts_bn)  # (B,G,M,C?), (B,G,3), (B,G,M)
        group_input_tokens = self.encoder(neighborhood)           # (B, G, encoder_dims)
        group_input_tokens = self.reduce_dim(group_input_tokens)  # (B, G, trans_dim)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos    = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos        = self.pos_embed(center)
        x   = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        feature_list_raw = self.blocks(x, pos)  # list of (B, G+1, trans_dim)
        cls_feat = self.norm(feature_list_raw[-1])[:, 0, :].contiguous()  # (B, trans_dim)
        patch_emb = self.norm(feature_list_raw[-1])[:, 1:, :].transpose(-1, -2).contiguous()  # (B,C,G)
        patch_centers = center.transpose(-1, -2).contiguous()
        return patch_emb, patch_centers, patch_idx, cls_feat

def get_loss(*args, **kwargs):
    return PT.get_loss(*args, **kwargs)
