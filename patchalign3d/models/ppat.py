# models/ppat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_redstone as rst
from einops import rearrange
import math
from pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

# ---- helper: FPS identical to PointTransformer.py ----

def fps(xyz_bN3, number):
    """
    xyz_bN3: (B, N, 3) float
    returns: (B, n, 3) with n = min(number, N)
    """
    _B, N, _ = xyz_bN3.shape
    n = min(number, N)
    idx = pointnet2_utils.furthest_point_sample(xyz_bN3, n)  # (B, n)
    fps_xyz = pointnet2_utils.gather_operation(
        xyz_bN3.transpose(1, 2).contiguous(), idx
    ).transpose(1, 2).contiguous()  # type: ignore[union-attr]
    return fps_xyz


# ---------------- core PPAT encoder ----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args, **kw):
        return self.fn(self.norm(x), *args, **kw)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., rel_pe=False):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.rel_pe = rel_pe
        if rel_pe:
            self.pe = nn.Sequential(nn.Conv2d(3, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x, centroid_delta):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv]
        pe = self.pe(centroid_delta) if self.rel_pe else 0
        dots = (q @ k.transpose(-1, -2) + pe) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = attn @ v
        return rearrange(out, 'b h n d -> b n (h d)')

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., rel_pe=False, fetch_idx=(3,7,11)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout, rel_pe)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]) for _ in range(depth)
        ])
        self.fetch_idx = set(fetch_idx)
        self.post_norm = nn.LayerNorm(dim)

    def forward_with_features(self, x, centroid_delta):
        """
        returns: list of (B, dim, G) at layers fetch_idx, and final x
        """
        feats = []
        for i, (attn, ff) in enumerate(self.layers):  # type: ignore[union-attr]
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
            if i in self.fetch_idx:
                x_n = self.post_norm(x)
                tok = x_n[:, 1:]                      # drop CLS
                feats.append(tok.transpose(1, 2).contiguous())  # (B, dim, G)
        return feats, x

    def forward(self, x, centroid_delta):
        for attn, ff in self.layers:  # type: ignore[union-attr]
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
        return x

class PointPatchTransformer(nn.Module):
    """
    PPAT tokenizer + transformer that can return [3,7,11] features like PointTransformer.
    """
    def __init__(self, dim, depth, heads, mlp_dim, sa_dim, patches, prad, nsamp,
                 in_dim=0, dim_head=64, rel_pe=False, patch_dropout=0, fetch_idx=(3,7,11)):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.patch_dropout = patch_dropout
        # SA expects conv in_channels = (3 xyz) + (in_dim features)
        self.sa = PointNetSetAbstraction(npoint=patches, radius=prad, nsample=nsamp,
                                         in_channel=in_dim + 3, mlp=[64, 64, sa_dim], group_all=False)
        self.lift = nn.Sequential(
            nn.Conv1d(sa_dim + 3, dim, 1),
            rst.Lambda(lambda x: torch.permute(x, [0, 2, 1])),
            nn.LayerNorm([dim])
        )
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 0.0, rel_pe, fetch_idx=fetch_idx)

    def encode_with_features(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        Returns:
          feat_list: [f1, f2, f3] each (B, dim, G)  from layers [3,7,11]
          ctr_b3g : (B, 3, G) patch centroids
        """
        self.sa.npoint = self.patches
        centroids, feat = self.sa(xyz, features)                    # (B,3,G), (B,sa_dim,G)
        x = self.lift(torch.cat([centroids, feat], dim=1))          # (B,G,dim)

        # add CLS, build relative deltas once
        x = rst.supercat([self.cls_token, x], dim=-2)               # (B,G+1,dim)
        centroids_plus = rst.supercat([centroids.new_zeros(1), centroids], dim=-1)   # (B,3,G+1)
        centroid_delta = centroids_plus.unsqueeze(-1) - centroids_plus.unsqueeze(-2)

        feat_list, _ = self.transformer.forward_with_features(x, centroid_delta)
        ctr_b3g = centroids_plus[:, :, 1:].contiguous()
        return feat_list, ctr_b3g

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        feat_list, _ = self.encode_with_features(xyz, features)
        # global pooled token proxy
        return sum([f.mean(dim=-1) for f in feat_list]) / len(feat_list)  # (B,dim)

# ------------- DGCNN_Propagation identical to PT -------------
def _best_gn_groups(C: int):
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1

class TorchKNN_Propagation(nn.Module):
    """
    Safe DGCNN-style propagation using torch.cdist on CPU (float64).
    Coords: (B,3,N), Feats: (B,C,N). Works for any trans_dim.
    """
    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.k = k

        # choose GN-friendly channels
        mid_raw = math.ceil((4.0 * dim) / 3.0)
        mid = ((mid_raw + 3) // 4) * 4
        g1 = _best_gn_groups(mid)
        g2 = _best_gn_groups(dim)

        self.layer1 = nn.Sequential(
            nn.Conv2d(2 * dim, mid, kernel_size=1, bias=False),
            nn.GroupNorm(g1, mid),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2 * mid, dim, kernel_size=1, bias=False),
            nn.GroupNorm(g2, dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @torch.no_grad()
    def _knn_idx_cpu64(self, ref_b3N: torch.Tensor, qry_b3N: torch.Tensor):
        """
        ref_b3N: (B,3,Nr)  neighbors drawn from here
        qry_b3N: (B,3,Nq)  queries
        returns idx: (B,k_eff,Nq) into ref_b3N’s last dim
        """
        _B, _, Nr = ref_b3N.shape
        _Nq = qry_b3N.shape[-1]
        k_eff = min(self.k, max(1, Nr))

        # guards against NaN/Inf
        if not torch.isfinite(ref_b3N).all():
            ref_b3N = torch.nan_to_num(ref_b3N, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(qry_b3N).all():
            qry_b3N = torch.nan_to_num(qry_b3N, nan=0.0, posinf=0.0, neginf=0.0)

        # (B,N,3) on CPU float64 for stability
        ref_bN3 = ref_b3N.transpose(1, 2).contiguous().cpu().double()
        qry_bN3 = qry_b3N.transpose(1, 2).contiguous().cpu().double()

        dist = torch.cdist(qry_bN3, ref_bN3)           # (B,Nq,Nr) CPU, float64
        _, idx = torch.topk(-dist, k=k_eff, dim=-1)    # (B,Nq,k_eff) CPU
        idx = idx.transpose(1, 2).contiguous()         # (B,k_eff,Nq) CPU long
        return idx.to(ref_b3N.device), k_eff           # move indices to GPU

    def _edge_feat(self, coor_k, x_k, coor_q, x_q):
        """
        coor_k,x_k: (B,3,Nk),(B,C,Nk)  source
        coor_q,x_q: (B,3,Nq),(B,C,Nq)  target
        -> (B,2C,Nq,k_eff)
        """
        B, C, Nk = x_k.size()
        Nq = x_q.size(2)
        idx, k_eff = self._knn_idx_cpu64(coor_k, coor_q)     # (B,k_eff,Nq) on GPU

        # gather neighbors from x_k
        idx_base = torch.arange(0, B, device=x_q.device).view(-1, 1, 1) * Nk
        idx_flat = (idx + idx_base).view(-1)                 # (B*k_eff*Nq,)
        xk_flat = x_k.transpose(2, 1).contiguous().view(B * Nk, C)
        neigh = xk_flat[idx_flat, :].view(B, k_eff, Nq, C).permute(0, 3, 2, 1).contiguous()  # (B,C,Nq,k)
        xq_rep = x_q.view(B, C, Nq, 1).expand(-1, -1, -1, k_eff)
        return torch.cat((neigh - xq_rep, xq_rep), dim=1)

    def forward(self, coor_k, feat_k, coor_q, feat_q):
        f = self._edge_feat(coor_k, feat_k, coor_q, feat_q)  # (B,2C,Nq,k)
        f = self.layer1(f).max(dim=-1, keepdim=False)[0]     # (B,mid,Nq)
        f2 = self._edge_feat(coor_q, f, coor_q, f)           # (B,2*mid,Nq,k)
        f2 = self.layer2(f2).max(dim=-1, keepdim=False)[0]   # (B,dim,Nq)
        return f2

# ------------- Full PPAT PartSeg with EXACT PT decoder -------------

class PPATEncoderOnly(nn.Module):
    """
    Encoder-only PPAT that exposes patch tokens + centers + membership indices.
    No decoder/propagation/head modules are included.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        _dim = cfg.trans_dim
        # Ensure SA sees at least 6 feature channels (OpenShape-style derived feat)
        in_dim_cfg = max(6, int(getattr(cfg, 'in_channel', 0)))
        self.ppat = PointPatchTransformer(
            dim=cfg.trans_dim,
            depth=cfg.depth,
            heads=cfg.num_heads,
            mlp_dim=getattr(cfg, 'mlp_dim', cfg.trans_dim * 4),
            sa_dim=getattr(cfg, 'sa_dim', 128),
            patches=getattr(cfg, 'patches', 512),
            prad=getattr(cfg, 'patch_radius', 0.2),
            nsamp=getattr(cfg, 'nsample', 32),
            in_dim=in_dim_cfg,
            dim_head=64,
            rel_pe=False,
            fetch_idx=getattr(cfg, 'fetch_idx', (3, 7, 11))
        )

    def _build_ppat_features(self, xyz, extras):
        """
        xyz: (B,3,N); extras: (B,C,N) or None
        Build 6-ch features = [xyz | const(0.4)] when extras is absent.
        """
        B, _, N = xyz.shape
        mode = getattr(self.cfg, 'feature_mode', 'xyz+const')
        if extras is None or extras.shape[1] == 0:
            feat3 = torch.ones_like(xyz) * 0.4
        else:
            C = extras.shape[1]
            if mode == 'xyz+rgb':
                feat3 = extras[:, :3, :] if C >= 3 else torch.cat(
                    [extras, torch.zeros(B, 3 - C, N, device=extras.device, dtype=extras.dtype)], dim=1)
            elif mode == 'xyz+normal':
                feat3 = extras[:, -3:, :] if C >= 6 else (extras[:, :3, :] if C >= 3 else
                        torch.cat([extras, torch.zeros(B, 3 - C, N, device=extras.device, dtype=extras.dtype)], dim=1))
            elif mode == 'xyz+first3':
                feat3 = extras[:, :3, :] if C >= 3 else torch.cat(
                    [extras, torch.zeros(B, 3 - C, N, device=extras.device, dtype=extras.dtype)], dim=1)
            else:
                feat3 = torch.ones(B, 3, N, device=xyz.device, dtype=xyz.dtype) * 0.4
        return torch.cat([xyz, feat3], dim=1)  # (B,6,N)

    def forward_patches(self, pts: torch.Tensor):
        """Return patch embeddings (B,C,G), centers (B,3,G), and membership idx (B,G,M)."""
        _B, C, N = pts.shape
        xyz = pts[:, :3, :].contiguous()
        extras = pts[:, 3:, :].contiguous() if C > 3 else None
        feats6 = self._build_ppat_features(xyz, extras)
        feat_list, ctr_b3g = self.ppat.encode_with_features(xyz, feats6)
        patch_emb = feat_list[-1]           # (B,dim,G)
        patch_centers = ctr_b3g             # (B,3,G)

        # Membership indices via KNN (xyz space)
        M = int(getattr(self.cfg, 'nsample', 32))
        try:
            knn = KNN(k=M, transpose_mode=True)
            _, idx = knn(xyz.transpose(1, 2).contiguous(), patch_centers.transpose(1, 2).contiguous())  # (B,G,M)
        except Exception:
            ref = xyz.transpose(1, 2).contiguous()           # (B,N,3)
            qry = patch_centers.transpose(1, 2).contiguous() # (B,G,3)
            dist = torch.cdist(qry, ref)
            idx = dist.topk(k=min(M, N), dim=-1, largest=False).indices
            if idx.shape[-1] < M:
                pad = idx[..., -1:].expand(*idx.shape[:-1], M - idx.shape[-1])
                idx = torch.cat([idx, pad], dim=-1)
        return patch_emb, patch_centers, idx

    # Loader focusing on encoder-only weights
    def load_model_from_ckpt(self, path):
        print(f"[PPATEncoderOnly] loading from {path}")
        ckpt = torch.load(path, map_location='cpu')
        raw = ckpt.get('state_dict', ckpt.get('model', ckpt))

        # Map common prefixes (DDP modules etc.)
        mapped = {}
        for k, v in raw.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[7:]
            if nk.startswith('point_encoder.'):
                nk = nk.replace('point_encoder.', 'ppat.')
            mapped[nk] = v

        sd = self.state_dict()
        to_load, shape_mismatch = {}, []
        # Accept only encoder keys
        enc_prefixes = ('ppat.sa.', 'ppat.lift.', 'ppat.transformer.', 'ppat.cls_token')
        for k, v in mapped.items():
            if k.startswith(enc_prefixes) or (k == 'ppat.cls_token'):
                if k in sd and sd[k].shape == v.shape:
                    to_load[k] = v
                elif k in sd:
                    shape_mismatch.append((k, tuple(v.shape), tuple(sd[k].shape)))
        res = self.load_state_dict(to_load, strict=False)
        print(f"[PPATEncoderOnly] loaded encoder tensors: {len(to_load)}")
        if res.missing_keys:
            print(f"[PPATEncoderOnly] MISSING ({len(res.missing_keys)}): {res.missing_keys[:10]}{' ...' if len(res.missing_keys)>10 else ''}")
        if res.unexpected_keys:
            print(f"[PPATEncoderOnly] UNEXPECTED IN CKPT ({len(res.unexpected_keys)}): {res.unexpected_keys[:10]}{' ...' if len(res.unexpected_keys)>10 else ''}")
        if shape_mismatch:
            print("[PPATEncoderOnly] SHAPE-MISMATCH:")
            for name, ck, md in shape_mismatch[:8]:
                print(f"  - {name}: ckpt{ck} vs model{md}")
class PPATPartSegPTDecoder(nn.Module):
    """
    PPAT encoder (tokens & centers) + EXACT PointTransformer decoder.
    Optional: use_proj_tokens -> apply projector to tokens before decoder (kept channel-consistent).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_parts = cfg.cls_dim
        self.num_classes = getattr(cfg, 'num_classes', 16)
        self.use_proj_tokens = bool(getattr(cfg, 'use_proj_tokens', False))

        dim = cfg.trans_dim
        # Ensure SA sees 9 channels total by default (3 xyz + 6 derived features)
        # OpenShape-style pretraining uses 6 extra channels even without true RGB.
        in_dim_cfg = max(6, int(getattr(cfg, 'in_channel', 0)))
        self.ppat = PointPatchTransformer(
            dim=cfg.trans_dim,
            depth=cfg.depth,
            heads=cfg.num_heads,
            mlp_dim=getattr(cfg, 'mlp_dim', cfg.trans_dim*4),
            sa_dim=getattr(cfg, 'sa_dim', 128),
            patches=getattr(cfg, 'patches', 512),
            prad=getattr(cfg, 'patch_radius', 0.2),
            nsamp=getattr(cfg, 'nsample', 32),
            in_dim=in_dim_cfg,
            dim_head=64,
            rel_pe=False,
            fetch_idx=getattr(cfg, 'fetch_idx', (3,7,11))
        )

        # projector from pretraining; if used on tokens, we adapt back to trans_dim if needed
        proj_out = int(getattr(cfg, 'out_channel', cfg.trans_dim))
        self.proj = nn.Linear(dim, proj_out)
        self._proj_back = None
        if self.use_proj_tokens and proj_out != dim:
            # keep EXACT decoder dims (trans_dim) -> adapt proj_out back to dim
            self._proj_back = nn.Conv1d(proj_out, dim, 1)

        # EXACT decoder modules (same in_channels as PT)
        self.propagation_2 = PointNetFeaturePropagation(in_channel= dim + 3,             mlp=[dim * 4, dim])
        self.propagation_1 = PointNetFeaturePropagation(in_channel= dim + 3,             mlp=[dim * 4, dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= dim + 3 + self.num_classes, mlp=[dim * 4, dim])
        D = cfg.trans_dim
        self.use_dgcnn = bool(getattr(cfg, 'use_dgcnn', False))  # default False; flip to True once stable
        self.dgcnn_pro_2 = TorchKNN_Propagation(dim=D, k=getattr(cfg, 'dgcnn_k', 16))
        self.dgcnn_pro_1 = TorchKNN_Propagation(dim=D, k=getattr(cfg, 'dgcnn_k', 16))

        # head (identical)
        self.conv1 = nn.Conv1d(dim, 128, 1)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_parts, 1)

    def get_loss(self):
        class _Loss(nn.Module):
            def forward(self, pred, target, trans_feat=None):
                return F.nll_loss(pred, target)
        return _Loss()

    def _maybe_project_feats(self, f_bgG):
        """
        f_bgG: (B, dim, G) tokens at some stage
        returns (B, dim, G) after optional projector (+adapter)
        """
        if not self.use_proj_tokens:
            return f_bgG
        # token-wise linear: (B,dim,G) -> (B,G,dim) -> proj -> (B,G,proj_out) -> (B,proj_out,G)
        _B, _C, _G = f_bgG.shape
        out = F.linear(f_bgG.transpose(1, 2), self.proj.weight, self.proj.bias).transpose(1, 2)
        if self._proj_back is not None:
            out = self._proj_back(out)
        return out
    def _build_ppat_features(self, xyz, extras):
        """
        xyz:    (B,3,N)
        extras: (B,Cextra,N) or None
        returns: (B,6,N) = [xyz (3) | feat3 (3)]
        """
        B, _, N = xyz.shape

        mode = getattr(self.cfg, 'feature_mode', 'xyz+const')  # 'xyz+rgb' | 'xyz+normal' | 'xyz+first3' | 'xyz+const'
        if extras is None or extras.shape[1] == 0:
            # No extra channels: follow OpenShape and synthesize a constant RGB (0.4)
            feat3 = torch.ones_like(xyz) * 0.4
        else:
            C = extras.shape[1]
            if mode == 'xyz+rgb':
                # take the first 3 channels of extras as RGB
                feat3 = extras[:, :3, :] if C >= 3 else torch.cat(
                    [extras, torch.zeros(B, 3-C, N, device=extras.device, dtype=extras.dtype)], dim=1)
            elif mode == 'xyz+normal':
                # if we have >=6, assume normals are the last 3; else use first 3
                feat3 = extras[:, -3:, :] if C >= 6 else (extras[:, :3, :] if C >= 3 else
                        torch.cat([extras, torch.zeros(B, 3-C, N, device=extras.device, dtype=extras.dtype)], dim=1))
            elif mode == 'xyz+first3':
                feat3 = extras[:, :3, :] if C >= 3 else torch.cat(
                    [extras, torch.zeros(B, 3-C, N, device=extras.device, dtype=extras.dtype)], dim=1)
            else:  # 'xyz+const' fallback
                feat3 = torch.ones(B, 3, N, device=xyz.device, dtype=xyz.dtype) * 0.4

        return torch.cat([xyz, feat3], dim=1)  # (B,6,N)
    def forward(self, pts, cls_onehot):
        B, C, N = pts.shape
        xyz = pts[:, :3, :].contiguous()
        extras = pts[:, 3:, :].contiguous() if C > 3 else None

        # Build 6-ch features = [xyz | feat3] to match PPAT pretraining
        feats6 = self._build_ppat_features(xyz, extras)           # (B,6,N)

        # ---- encoder: return [3,7,11] token maps and patch centers ----
        feat_list, ctr_b3g = self.ppat.encode_with_features(xyz, feats6)

        # (optional) project tokens then adapt back to trans_dim so decoder shapes match
        feat_list = [self._maybe_project_feats(f) for f in feat_list]

        # ---- EXACT PointTransformer decoder below (unchanged) ----
        xyz_bN3 = xyz.transpose(1, 2).contiguous()                       # (B,N,3)
        center_level_0 = xyz
        center_level_1 = fps(xyz_bN3, 512).transpose(1, 2).contiguous()
        center_level_2 = fps(xyz_bN3, 256).transpose(1, 2).contiguous()
        center_level_3 = ctr_b3g

        cls_oh = cls_onehot.view(B, self.num_classes, 1).repeat(1, 1, N)
        f_level_0 = torch.cat([cls_oh, center_level_0], 1)
        f_level_1 = center_level_1
        f_level_2 = center_level_2

        f_level_3 = feat_list[2]
        if not hasattr(self, "_dbg_once"):
            print("[dbg] shapes:",
                "f3", tuple(f_level_3.shape),
                "c3", tuple(center_level_3.shape),
                "f2", tuple(f_level_2.shape),
                "c2", tuple(center_level_2.shape),
                "f1", tuple(f_level_1.shape),
                "c1", tuple(center_level_1.shape))
            # sanity: all coords must be (B,3,N), feats (B,D,N)
            for name, t in [("c3", center_level_3), ("c2", center_level_2), ("c1", center_level_1),
                            ("f3", f_level_3), ("f2", f_level_2), ("f1", f_level_1)]:
                assert t.isfinite().all(), f"{name} has NaN/Inf"
            self._dbg_once = True
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feat_list[1])
        f_level_1 = self.propagation_1(center_level_1, center_level_2, f_level_1, feat_list[0])



        if not hasattr(self, "_dbg_postfp"):
            print("[post-FP] f2", tuple(f_level_2.shape), "f1", tuple(f_level_1.shape), "dim", self.cfg.trans_dim)
            self._dbg_postfp = True

        ok_channels = (f_level_2.shape[1] == self.cfg.trans_dim) and (f_level_1.shape[1] == self.cfg.trans_dim)
        if self.use_dgcnn and ok_channels:
            # bottom-up refinement with DGCNN (safe Torch KNN)
            f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)  # (B,dim,256)
            f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)  # (B,dim,512)
        elif self.use_dgcnn and not ok_channels:
            if not hasattr(self, "_warn_once"):
                print("[warn] Skipping DGCNN: FP channels are",
                    f"f2={f_level_2.shape[1]}, f1={f_level_1.shape[1]} (expect {self.cfg.trans_dim}).")
                self._warn_once = True


        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)

        f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        feat = F.relu(self.bn1(self.conv1(f_level_0)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1).permute(0, 2, 1)
        return x, f_level_3

    def forward_patches(self, pts: torch.Tensor):
        """
        Encoder-only interface to mirror PointTransformer_patched.forward_patches.
        Returns patch embeddings, centers, and per-patch membership indices.

        Args:
            pts: (B, C, N) with C>=3: [xyz | extras]
        Returns:
            patch_emb    : (B, trans_dim, G)
            patch_centers: (B, 3, G)
            patch_idx    : (B, G, M)  indices of point-membership per patch
        """
        _B, C, N = pts.shape
        xyz = pts[:, :3, :].contiguous()
        extras = pts[:, 3:, :].contiguous() if C > 3 else None

        feats6 = self._build_ppat_features(xyz, extras)  # (B,6,N)
        feat_list, ctr_b3g = self.ppat.encode_with_features(xyz, feats6)
        feat_list = [self._maybe_project_feats(f) for f in feat_list]

        patch_emb = feat_list[-1]        # (B, dim, G)
        patch_centers = ctr_b3g          # (B, 3, G)

        # Build membership indices via KNN in xyz space
        M = int(getattr(self.cfg, 'nsample', 32))
        try:
            knn = KNN(k=M, transpose_mode=True)
            # KNN expects (B,N,3) and (B,G,3)
            _, idx = knn(xyz.transpose(1, 2).contiguous(),
                         patch_centers.transpose(1, 2).contiguous())  # (B,G,M)
        except Exception:
            # Safe torch.cdist fallback
            ref = xyz.transpose(1, 2).contiguous()              # (B,N,3)
            qry = patch_centers.transpose(1, 2).contiguous()    # (B,G,3)
            # dist: (B,G,N); take top-k nearest (smallest)
            dist = torch.cdist(qry, ref)                        # (B,G,N)
            idx = dist.topk(k=min(M, N), dim=-1, largest=False).indices  # (B,G,M')
            # If M>N, pad with last index for shape consistency
            if idx.shape[-1] < M:
                pad = idx[..., -1:].expand(*idx.shape[:-1], M - idx.shape[-1])
                idx = torch.cat([idx, pad], dim=-1)

        return patch_emb, patch_centers, idx

    # --------- robust encoder/proj loader (prints loaded/missing) ----------
    def load_model_from_ckpt(self, path):
        import re
        print(f"[PPATPartSegPTDecoder] loading from {path}")
        ckpt = torch.load(path, map_location='cpu')
        raw = ckpt.get('state_dict', ckpt.get('model', ckpt))

        # 1) Map common prefixes (DDP etc.)
        mapped = {}
        for k, v in raw.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[7:]
            if nk.startswith('point_encoder.'):
                nk = nk.replace('point_encoder.', 'ppat.')
            mapped[nk] = v

        # 2) Try to locate projector weights in ckpt under common aliases
        #    We look for a *.weight tensor with shape [out_dim, in_dim]
        proj_weight_key = None
        proj_bias_key = None
        alias_patterns = [
            r'^proj\.weight$', r'^proj\.bias$',
            r'^projector\.weight$', r'^projector\.bias$',
            r'^projection\.weight$', r'^projection\.bias$',
            r'^mlp_head\.weight$', r'^mlp_head\.bias$',
            r'^head\.weight$', r'^head\.bias$'
        ]
        # build quick lookup
        keys = list(mapped.keys())
        def find_key(patterns, is_weight=True):
            for pat in patterns:
                if (is_weight and pat.endswith('weight$')) or ((not is_weight) and pat.endswith('bias$')):
                    rx = re.compile(pat)
                    for k in keys:
                        if rx.match(k):
                            return k
            return None

        proj_weight_key = find_key(alias_patterns, is_weight=True)
        proj_bias_key   = find_key(alias_patterns, is_weight=False)

        # 3) If found projector in ckpt, reconcile shapes with current model
        loaded_proj = False
        if proj_weight_key is not None:
            W = mapped[proj_weight_key]
            B = mapped.get(proj_bias_key, None)
            out_ck, in_ck = W.shape[0], W.shape[1]

            # Rebuild self.proj to match ckpt if needed
            if self.proj.weight.shape != W.shape or (self.proj.bias is None) != (B is None):
                print(f"[PPATPartSegPTDecoder] resetting proj to match ckpt: ({out_ck},{in_ck})")
                # re-create projector module with ckpt dims
                new_proj = torch.nn.Linear(in_ck, out_ck, bias=(B is not None))
                # move to same device as model
                new_proj.to(next(self.parameters()).device)
                self.proj = new_proj  # register
                # (Re)build adapter if we use projected tokens and proj_out != trans_dim
                if self.use_proj_tokens:
                    if out_ck != self.cfg.trans_dim:
                        self._proj_back = torch.nn.Conv1d(out_ck, self.cfg.trans_dim, 1).to(next(self.parameters()).device)
                    else:
                        self._proj_back = None

            # Load projector weights
            with torch.no_grad():
                self.proj.weight.copy_(W)
                if (self.proj.bias is not None) and (B is not None):
                    self.proj.bias.copy_(B)
            loaded_proj = True
        else:
            print("[PPATPartSegPTDecoder] projector not found in ckpt (searched for proj/projector/projection/mlp_head/head).")

        # 4) Build encoder load dict (ppat.* and direct internals: sa/lift/transformer)
        sd = self.state_dict()
        to_load, shape_mismatch = {}, []

        for k, v in mapped.items():
            # already handled projector above
            if k in (proj_weight_key, proj_bias_key):
                continue
            # try exact match
            if k in sd:
                if sd[k].shape == v.shape:
                    to_load[k] = v
                else:
                    shape_mismatch.append((k, tuple(v.shape), tuple(sd[k].shape)))
                continue
            # allow bare encoder internals -> ppat.*
            if k.startswith(('sa.', 'lift.', 'transformer.', 'cls_token')):
                kk = 'ppat.' + k
                if kk in sd and sd[kk].shape == v.shape:
                    to_load[kk] = v
                elif kk in sd:
                    shape_mismatch.append((kk, tuple(v.shape), tuple(sd[kk].shape)))

        # 5) Load encoder tensors
        res = self.load_state_dict(to_load, strict=False)

        # 6) Report
        print(f"[PPATPartSegPTDecoder] loaded encoder tensors: {len(to_load)}  |  projector loaded: {loaded_proj}")
        if res.missing_keys:
            print(f"[PPATPartSegPTDecoder] MISSING ({len(res.missing_keys)}): {res.missing_keys[:10]}{' ...' if len(res.missing_keys)>10 else ''}")
        if res.unexpected_keys:
            print(f"[PPATPartSegPTDecoder] UNEXPECTED IN CKPT ({len(res.unexpected_keys)}): {res.unexpected_keys[:10]}{' ...' if len(res.unexpected_keys)>10 else ''}")
        if shape_mismatch:
            print("[PPATPartSegPTDecoder] SHAPE-MISMATCH:")
            for name, ck, md in shape_mismatch[:8]:
                print(f"  - {name}: ckpt{ck} vs model{md}")
            if len(shape_mismatch) > 8:
                print(f"  ... and {len(shape_mismatch)-8} more")
        if loaded_proj:
            print(f"[PPATPartSegPTDecoder] proj.weight -> {tuple(self.proj.weight.shape)}  proj.bias -> {None if self.proj.bias is None else tuple(self.proj.bias.shape)}")
            if self.use_proj_tokens:
                print(f"[PPATPartSegPTDecoder] token projection enabled; adapter = "
                    f"{'None (proj_out==trans_dim)' if self._proj_back is None else f'1x1 Conv: {self._proj_back.in_channels}->{self._proj_back.out_channels}'}")

# ---- factory for train_partseg compatibility ----
def get_model(config, **kwargs):
    """Return encoder-only PPAT by default for patch-based training."""
    return PPATEncoderOnly(config)

class get_loss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, target, trans_feat):
        return F.nll_loss(pred, target)
