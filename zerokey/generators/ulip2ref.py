"""ULIP2 reference-view baseline generator for 3D keypoint detection."""

from pathlib import Path
from types import SimpleNamespace
from functools import cached_property
from typing import Any

from einops import rearrange
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds
import torch
import numpy as np
import numpy.typing as npt
from io import BytesIO

from zerokey.io.kpnet import RefIO
from zerokey.generators.patchalign3d import PatchAlign3DGenerator, match_reference_features

import ULIP.models.ULIP_models as ulip_models

ULIP2_CKPT_PATH = Path(__file__).parent.parent.parent / "ULIP" / "models" / "pointbert" / "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"

def fps_indices(xyz: npt.NDArray[np.floating[Any]], k: int, *, seed: int = 0) -> npt.NDArray[np.int64]:
    """Furthest Point Sampling on CPU (numpy). Returns indices of size k.
    xyz: (N,3)
    If N <= k, returns np.arange(N).
    """
    N = xyz.shape[0]
    if N <= k:
        return np.arange(N, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idxs = np.empty(k, dtype=np.int64)
    # Greedy farthest-point selection: start from a random seed point,
    # then iteratively pick the point with maximum distance to all
    # previously selected points. dists[j] tracks the minimum squared
    # distance from point j to any selected point so far.
    idxs[0] = int(rng.randint(0, N))
    dists = np.full(N, np.inf, dtype=np.float64)
    last = xyz[idxs[0]][None, :]
    dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    for i in range(1, k):
        idxs[i] = int(np.argmax(dists))  # point farthest from current selection
        last = xyz[idxs[i]][None, :]
        dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    return idxs


def resample_points(xyzf: npt.NDArray[np.floating[Any]], npoints: int, *, seed: int = 0) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]:
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

def load_ulip2_model(device: str | torch.device = "cuda") -> Any:
    """
    Load ULIP2 PointBERT model for point cloud feature extraction.
    Uses hardcoded checkpoint path from ULIP/models/pointbert/

    Args:
        device: Device to load model on.

    Returns:
        ULIP2 model in eval mode.
    """
    from collections import OrderedDict

    print("Loading ULIP2 PointBERT model...")
    model = ulip_models.ULIP2_PointBERT_Colored(args=SimpleNamespace())
    model.eval()
    model = model.to(device)

    # Load checkpoint using same pattern as test_zeroshot_3d_ulip2
    if not ULIP2_CKPT_PATH.exists():
        raise FileNotFoundError(f"ULIP2 checkpoint not found at {ULIP2_CKPT_PATH}")

    print(f"Loading checkpoint from {ULIP2_CKPT_PATH}")
    ckpt = torch.load(ULIP2_CKPT_PATH, map_location='cpu')
    # Strip 'module.' prefix from keys: checkpoints saved with nn.DataParallel
    # wrap all keys as 'module.layer_name', but we load into a bare model.
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    model.load_state_dict(state_dict, strict=False)
    print("✓ ULIP2 checkpoint loaded successfully")

    return model


def prepare_pointcloud_for_ulip2(mesh: Meshes, npoints: int = 10000) -> tuple[torch.Tensor, npt.NDArray[np.int64]]:
    """
    Convert mesh to point cloud format expected by ULIP2.
    Reuses ULIP data preprocessing utilities.

    Args:
        mesh: PyTorch3D mesh object
        npoints: Number of points to sample (default 10000 for ULIP2)

    Returns:
        Point cloud tensor in format (1, C, N) where C=6 (xyz+rgb)
    """
    from ULIP.data.dataset_3d import pc_normalize

    # Get vertices
    verts_packed = mesh.verts_packed()
    assert verts_packed is not None
    vertices = verts_packed.cpu().numpy()  # (N, 3)

    vertices, sample_to_input_idx = resample_points(vertices, npoints)

    # Normalize point cloud using ULIP's pc_normalize
    vertices = pc_normalize(vertices)

    # Add RGB (default gray color 0.4, same as ULIP dataset)
    rgb = np.ones_like(vertices) * 0.4
    point_cloud = np.concatenate([vertices, rgb], axis=1)  # (N, 6)

    # Convert to tensor and reshape to (1, C, N) format expected by ULIP2
    pc_tensor = torch.from_numpy(point_cloud).float()
    pc_tensor = pc_tensor.transpose(0, 1).unsqueeze(0)  # (1, 6, N)

    return pc_tensor, sample_to_input_idx


def extract_pc_features_ulip2(model: Any, point_cloud: torch.Tensor, device: str | torch.device = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract point cloud features using ULIP2 model.

    Args:
        model: ULIP2 model
        point_cloud: Point cloud tensor in format (1, C, N)
        device: Device to run on

    Returns:
        Point cloud features tensor
    """
    model.eval()
    point_cloud = point_cloud.to(device)

    with torch.no_grad():
        pc_embed, center, patch_features = model.encode_pc(point_cloud)
        # Normalize features
        pc_embed = pc_embed / pc_embed.norm(dim=-1, keepdim=True)

    return pc_embed, center, patch_features


class ULIP2RefIO(RefIO):
    """ULIP2-specific reference I/O (inherits all behavior from RefIO)."""
    pass


class ULIP2RefGenerator(PatchAlign3DGenerator):
    """Reference-view generator using ULIP2 point cloud features.

    Extracts global and patch-level features from point clouds using ULIP2
    (PointBERT backbone), then matches keypoints by comparing reference
    features against patch embeddings via cosine similarity.

    Attributes:
        KPIO: Uses ULIP2RefIO for reference feature accumulation.
        Multimodal: ULIP2_WITH_OPENCLIP model loaded lazily.
    """
    KPIO = ULIP2RefIO
    Multimodal = ulip_models.ULIP2_WITH_OPENCLIP
    io: ULIP2RefIO

    @cached_property
    def multimodal(self) -> Any:
        """Lazily load the ULIP2 PointBERT model."""
        return load_ulip2_model(device=self.device)

    def process_kp_list(self, mesh: Meshes, fragments: Any, R: CamerasBase, T: Any, images: torch.Tensor, kp_list: dict[Any, Any], class_title: str, mesh_id: str, prompt_idx: int | slice = 0) -> dict[frozenset[Any], Pointclouds]:
        """Extract ULIP2 features, collect references, and match keypoints.

        Converts the mesh to a normalized point cloud, extracts ULIP2 patch
        features, then either accumulates reference features or matches each
        keypoint against stored references using cosine similarity.
        """
        # Extract point cloud features using ULIP2
        pc_tensor, idx = prepare_pointcloud_for_ulip2(mesh, npoints=10000)
        # Transpose from channels-first (1, C, N) to points-last (1, N, C) layout
        # expected by ULIP2's encode_pc, where C=6 (xyz+rgb) and N=10000 points.
        pc_tensor = rearrange(pc_tensor, 'b c n -> b n c')
        _, center, pc_features = extract_pc_features_ulip2(self.multimodal, pc_tensor, device=self.device)

        # Serialize ULIP2 outputs as npz for consumption by RefIO.sample_reference_view.
        # squeeze(0) removes the batch dim: center [1, G, 3] -> [G, 3], etc.
        with BytesIO() as in_buffer:
            np.savez(in_buffer,
                points_sampled=pc_tensor[..., :3].cpu().squeeze(0).numpy(),
                patch_centers=center.cpu().squeeze(0).numpy(),
                sample_to_input_idx=idx,
                patch_emb=pc_features.cpu().squeeze(0).numpy()
            )
            npz_file = in_buffer.getvalue()

        with self.io.sample_reference_view(npz_file, kp_list, mesh, class_title) as used_as_reference:
            if not used_as_reference:
                raise LookupError("No reference view found")
            d = match_reference_features(self.io, npz_file, kp_list, class_title, self.device)

        return {frozenset(kp_list.keys()): self.backproject_kps(mesh, fragments, R, T, (d,))}

