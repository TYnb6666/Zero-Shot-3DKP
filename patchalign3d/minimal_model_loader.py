"""
Minimal loader for the two-stage Find3D segmentation model.

- Reconstructs only the encoder + lightweight heads actually used during training
  (projection to text space + optional temperature/bias), and omits unused
  propagation/decoder modules.
- Loads weights from a checkpoint like:
    model_ckpt/2stagemodel.pt  (or set $POINTBERT_CKPT)
- Prints a parameter count summary.

Run:
  python minimal_model_loader.py \
    --ckpt $POINTBERT_CKPT

Optional flags:
  --device cpu|cuda   (default: auto)
  --color true|false  (auto-inferred from checkpoint if possible)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure we can import the minimal encoder (encoder-only implementation)
# The file lives in segmentation/models/point_encoder.py
try:
    from models.point_encoder import PointTransformer as EncoderOnly
    from models.point_encoder import build_default_encoder
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import models.point_encoder. Make sure you run this script from \n"
        "Point-BERT/segmentation or that 'segmentation/models' is on PYTHONPATH."
    ) from e


class PatchToTextProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb: torch.Tensor) -> torch.Tensor:
        # patch_emb: (B, C, G) -> (B, G, C)
        x = patch_emb.transpose(1, 2)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        # back to (B, C, G)
        return x.transpose(1, 2)


class LearnableTemp(nn.Module):
    """Inverse-temperature scale with exp-parameterization by default."""

    def __init__(self, init_tau: float = 0.07, mode: str = "exp", init_linear: float = 2.302585093, max_scale: float = 100.0):
        super().__init__()
        self.mode = mode
        self.max_scale = float(max_scale)
        if mode == "linear":
            self.scale = nn.Parameter(torch.tensor(float(init_linear), dtype=torch.float32))
            self.log_scale = None
        else:
            init_scale = 1.0 / max(init_tau, 1e-6)
            self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
            self.scale = None

    def forward(self) -> torch.Tensor:
        if self.mode == "linear":
            assert self.scale is not None
            return torch.clamp(self.scale, max=self.max_scale)
        assert self.log_scale is not None
        return self.log_scale.exp().clamp(max=self.max_scale)


@dataclass
class MinimalStage2Config:
    # Encoder defaults per training
    trans_dim: int = 384
    depth: int = 12
    drop_path_rate: float = 0.1
    num_heads: int = 6
    group_size: int = 32
    num_group: int = 128
    encoder_dims: int = 256
    color: bool = False  # Stage-2 typically trained without color
    # Text head
    text_dim: int = 512  # Will be auto-inferred from checkpoint when possible


class MinimalStage2Model(nn.Module):
    """Encoder-only + projection + temperature (no decoder/propagation)."""

    def __init__(self, cfg: MinimalStage2Config):
        super().__init__()
        self.cfg = cfg
        # Build encoder-only model that mirrors training-time parameter names
        self.point_encoder: EncoderOnly = build_default_encoder(
            trans_dim=cfg.trans_dim,
            depth=cfg.depth,
            drop_path_rate=cfg.drop_path_rate,
            num_heads=cfg.num_heads,
            group_size=cfg.group_size,
            num_group=cfg.num_group,
            encoder_dims=cfg.encoder_dims,
            color=cfg.color,
        )
        # Heads used at training time
        self.proj = PatchToTextProj(in_dim=cfg.trans_dim, out_dim=cfg.text_dim)
        self.temp_module = LearnableTemp(init_tau=0.07, mode="exp")
        # Optional bias module (present in some runs)
        self.bias_module: Optional[nn.Module] = None

    @torch.no_grad()
    def load_from_checkpoint(self, ckpt_path: str) -> Dict[str, int]:
        """Load encoder + heads from a training checkpoint.

        Returns counts of loaded items for quick sanity checking.
        """
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load once to inspect structure
        obj = torch.load(str(path), map_location="cpu")

        # 1) Encoder (robust internal loader handles nested dicts and prefixes)
        enc_stats = self.point_encoder.load_weights(str(path))

        # 2) Heads
        loaded_proj = loaded_temp = loaded_bias = 0
        if isinstance(obj, dict):
            if "proj" in obj and isinstance(obj["proj"], dict):
                try:
                    self.proj.load_state_dict(obj["proj"], strict=True)
                    loaded_proj = sum(p.numel() for p in self.proj.parameters())
                except Exception:
                    # Fallback to non-strict load
                    self.proj.load_state_dict(obj["proj"], strict=False)
                    loaded_proj = sum(p.numel() for p in self.proj.parameters())

            if "temp_module" in obj and isinstance(obj["temp_module"], dict):
                try:
                    self.temp_module.load_state_dict(obj["temp_module"], strict=True)
                    loaded_temp = sum(p.numel() for p in self.temp_module.parameters())
                except Exception:
                    self.temp_module.load_state_dict(obj["temp_module"], strict=False)
                    loaded_temp = sum(p.numel() for p in self.temp_module.parameters())

            # Optional bias module
            if "bias_module" in obj and isinstance(obj["bias_module"], dict):
                # Create lazily if present
                self.bias_module = nn.Identity()
                try:
                    # For compatibility, allow identity (no params) if shapes don't match
                    state = obj["bias_module"]
                    # If state is empty, skip
                    if state:
                        # Try a single-parameter bias module
                        self.bias_module = nn.Sequential(nn.Identity())
                    loaded_bias = 0
                except Exception:
                    self.bias_module = nn.Identity()
                    loaded_bias = 0

        return {
            "encoder_loaded": int(enc_stats.get("loaded", 0)),
            "encoder_missing": int(enc_stats.get("missing", 0)),
            "encoder_unexpected": int(enc_stats.get("unexpected", 0)),
            "proj_params": loaded_proj,
            "temp_params": loaded_temp,
            "bias_params": loaded_bias,
        }

    def forward_patches(self, pts: torch.Tensor):
        """Return encoder patch features and centers.

        pts: (B, C, N) tensor (xyz | [rgb]) with C>=3
        Returns: patch_emb (B, trans_dim, G), patch_centers (B, 3, G), patch_idx (B, G, M)
        """
        return self.point_encoder.forward_patches(pts)


def _infer_color_from_ckpt(ckpt_obj: object, default: bool = False) -> bool:
    """Try to infer whether the encoder was trained with color channels.

    Looks for a conv weight with shape (128, in_ch, 1) under encoder.first_conv.0.weight.
    """
    try:
        sd = None
        if isinstance(ckpt_obj, dict):
            sd = ckpt_obj.get("model") or ckpt_obj.get("state_dict") or ckpt_obj
        if isinstance(sd, dict):
            # Normalize prefixes
            for k, v in sd.items():
                name = k
                if name.startswith("module."):
                    name = name[len("module."):]
                if name.startswith("point_encoder."):
                    name = name[len("point_encoder."):]
                if name.endswith("encoder.first_conv.0.weight") and hasattr(v, "shape"):
                    in_ch = int(v.shape[1])
                    return in_ch == 6
    except Exception:
        pass
    return bool(default)


def _infer_text_dim_from_ckpt(ckpt_obj: object, default: int = 512) -> int:
    try:
        if isinstance(ckpt_obj, dict):
            proj_sd = ckpt_obj.get("proj")
            if isinstance(proj_sd, dict):
                for k, v in proj_sd.items():
                    if k.endswith("weight") and hasattr(v, "shape") and len(v.shape) == 2:
                        # nn.Linear(out_dim, in_dim) in state_dict order for weight
                        out_dim, _in_dim = int(v.shape[0]), int(v.shape[1])
                        return out_dim
    except Exception:
        pass
    return int(default)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Load minimal 2-stage model and report parameter counts")
    ap.add_argument("--ckpt", type=str, default=str(Path(__file__).resolve().parent / "model_ckpt/2stagemodel.pt"))
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--color", type=str, default="auto", choices=["auto", "true", "false"], help="Use color channels (6D input). Default: auto-infer from checkpoint")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")

    # Infer color and text_dim when not provided
    if args.color == "auto":
        color = _infer_color_from_ckpt(ckpt_obj, default=False)
    else:
        color = (args.color.lower() == "true")
    text_dim = _infer_text_dim_from_ckpt(ckpt_obj, default=512)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Build minimal model
    cfg = MinimalStage2Config(color=color, text_dim=text_dim)
    model = MinimalStage2Model(cfg)
    model.to(device)

    # Load weights
    load_stats = model.load_from_checkpoint(str(ckpt_path))

    # Count parameters
    enc_params = count_params(model.point_encoder)
    proj_params = count_params(model.proj)
    temp_params = count_params(model.temp_module)
    bias_params = count_params(model.bias_module) if model.bias_module is not None else 0
    total_params = enc_params + proj_params + temp_params + bias_params

    print("Minimal Stage-2 Model Summary (encoder-only + heads)")
    print(f"- Checkpoint: {ckpt_path}")
    print(f"- Device    : {device}- Color     : {color}")
    print(f"- Text dim  : {text_dim}")
    print("Load stats:")
    for k, v in load_stats.items():
        print(f"  {k}: {v}")
    print("Parameter counts:")
    print(f"  encoder : {enc_params:,}")
    print(f"  proj    : {proj_params:,}")
    print(f"  temp    : {temp_params:,}")
    if bias_params:
        print(f"  bias    : {bias_params:,}")
    print(f"  TOTAL   : {total_params:,}")


if __name__ == "__main__":
    main()

