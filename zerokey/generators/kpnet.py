"""Zero-shot 3D keypoint detection pipeline using multi-view MLLM queries and HDBSCAN clustering."""

import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import List, Union, Any, Generic, ClassVar, TypeVar, cast, get_args
from collections import defaultdict, Counter
from pathlib import Path
from functools import cached_property, partialmethod
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds
from fast_hdbscan import HDBSCAN
from zerokey.rendering import views_from_model, RenderO3D, get_depth_point_cloud, debug_enabled
from zerokey.models import GPT4o, Molmo
from zerokey._detection import KeypointDetectionMixin
from zerokey.features.pointnext_surface import load_surface_features, surface_feature_path
from zerokey.io.kpnet import KPNetIO
from kp_utils import sample_view_points

# TypeVars for Generic KPNetGenerator
# _IO: I/O handler type, bound to KPNetIO for common I/O interface.
#      KPIO ClassVar is auto-derived from the type parameter via __init_subclass__.
#      e.g.: class MyGenerator(KPNetGenerator[MyIO, Molmo]): ...
#
# _M: Multimodal model type. Unbound TypeVar because Multimodal implementations
#     vary widely (Molmo, PatchExplorer, RedCircle, etc.) with incompatible interfaces.
#     Multimodal ClassVar is auto-derived from the type parameter via __init_subclass__.
#     Subclasses that use non-Molmo Multimodal classes override the methods that
#     access `multimodal` (detect_kps, backproject_kps, init_kps_batch).
_IO = TypeVar('_IO', bound=KPNetIO, covariant=True)
_M = TypeVar('_M', covariant=True)


class KPNetGenerator(KeypointDetectionMixin, RenderO3D, Generic[_IO, _M]):
    """
    Main generator class for Zero-Shot 3D Keypoint detection.

    It orchestrates the pipeline of:
    1. Rendering multiple views of a 3D mesh.
    2. Using a Multimodal LLM (e.g., Molmo) to detect 2D keypoints in those views.
    3. Backprojecting the 2D detections into 3D space using depth information.
    4. Aggregating multi-view 3D detections via clustering (HDBSCAN) to find stable 3D keypoints.

    Type Parameters:
        _IO: I/O handler type (default: KPNetIO). Auto-derived into KPIO ClassVar.
        _M: Multimodal model type. Auto-derived into Multimodal ClassVar.
             Explicit ClassVar assignments in the class body take precedence.
    """
    # Class attributes for Multimodal model and I/O handler types.
    # Subclasses override these to specify different implementations.
    Multimodal: ClassVar[type[Any]] = Molmo
    KPIO: ClassVar[type[KPNetIO]] = KPNetIO
    views_from_model = partialmethod(views_from_model)
    sample_view_points = staticmethod(sample_view_points)

    @cached_property
    def gpt(self) -> GPT4o:
        """Lazy-initializing GPT-4o instance (only needed for get_prompts)."""
        return GPT4o()

    @cached_property
    def multimodal(self) -> _M:
        """Lazy-initializing multimodal model instance."""
        return self.Multimodal()

    @cached_property
    def io(self) -> _IO:
        """I/O handler instance, created from KPIO class attribute."""
        return cast(_IO, self.KPIO(self.log_dir / self.expname))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-derive KPIO and Multimodal ClassVars from type parameters.

        Inspects ``__orig_bases__`` for direct ``KPNetGenerator[IO, M]``
        parameterizations and sets ``cls.KPIO`` / ``cls.Multimodal`` unless
        the subclass already defines them explicitly in its class body.
        """
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, '__orig_bases__', ()):
            origin = getattr(base, '__origin__', None)
            if origin is not KPNetGenerator:
                continue
            args = get_args(base)
            if 'KPIO' not in cls.__dict__ and len(args) >= 1 and not isinstance(args[0], TypeVar):
                cls.KPIO = args[0]
            if 'Multimodal' not in cls.__dict__ and len(args) >= 2 and not isinstance(args[1], TypeVar):
                cls.Multimodal = args[1]

    def __init__(self, log_dir: Union[str, os.PathLike] = Path(), expname: str = f'{type(Multimodal).__name__}PTS',
                 res: int | tuple[int, int] = 512, scale: int | float = 2,
                 use_molmo_vit_features: bool = False, molmo_vit_feature_dim: int = 64,
                 molmo_vit_feature_scale: float = 0.1,
                 molmo_vit_feature_file: str = 'molmo_vit_features.pt',
                 molmo_vit_feature_expname: str | None = None,
                 molmo_2d_expname: str | None = None,
                 use_pointnext_surface_features: bool = False,
                 pointnext_surface_feature_dir: str | os.PathLike | None = None,
                 pointnext_feature_dim: int = 16,
                 pointnext_feature_scale: float = 0.1,
                 pointnext_voxel_size: float = 0.01,
                 pointnext_max_disk_diameter: float = 0.1,
                 pointnext_max_disk_depth_range: float = 0.03,
                 pointnext_max_disk_depth_ratio: float = 0.75,
                 pointnext_min_disk_points: int = 50,
                 pointnext_max_surface_dist: float | None = None):
        """Initialize the generator with output paths, rendering config, and viewpoints.

        Args:
            log_dir: Root output directory for experiment results.
            expname: Experiment name used as subdirectory under log_dir.
            res: Base render resolution (pixels). Actual rendering is at res*scale.
            scale: Upscale factor; renders at res*scale then downscales for anti-aliasing.
            use_molmo_vit_features: If True, append sampled cached Molmo ViT
                features to the HDBSCAN clustering space.
            molmo_vit_feature_dim: Random-projection output dimension before clustering.
            molmo_vit_feature_scale: Scale applied to standardized reduced features.
            molmo_vit_feature_file: Per-mesh feature-cache filename.
            molmo_vit_feature_expname: Optional source experiment for cached features.
            molmo_2d_expname: Optional source experiment for cached raw Molmo 2D detections.
            use_pointnext_surface_features: If True, snap valid backprojected
                candidates to saved PointNeXt surface anchors before clustering.
            pointnext_surface_feature_dir: Root containing category/mesh_id.pt
                PointNeXt surface-anchor artifacts.
            pointnext_feature_dim: Random-projection output dimension for
                PointNeXt features before HDBSCAN.
            pointnext_feature_scale: Scale applied to standardized projected
                PointNeXt features in the HDBSCAN space.
            pointnext_voxel_size: Voxel size for downsampling snapped normal-view
                candidates before HDBSCAN. Set <= 0 to disable.
            pointnext_max_disk_diameter: Reject each view/color disk if its raw
                valid 3D candidates have a robust diameter larger than this
                value in ZeroKey coordinates. Set <= 0 to disable.
            pointnext_max_disk_depth_range: Reject each view/color disk if its
                robust depth range along the viewing direction is larger than
                this value. This targets depth jumps and discontinuity edges.
                Set <= 0 to disable.
            pointnext_max_disk_depth_ratio: Reject each view/color disk if its
                robust depth range divided by robust lateral diameter is larger
                than this value. Set <= 0 to disable.
            pointnext_min_disk_points: Minimum valid raw candidates required to
                keep a view/color disk for PointNeXt snapping.
            pointnext_max_surface_dist: Optional maximum distance from raw
                candidate to nearest saved surface anchor after snapping.
        """
        self.log_dir = Path(log_dir)
        self.expname = expname
        self.dist = 1
        self.res = res
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vis = debug_enabled()
        self.views = self.sample_view_points(self.dist, partition=3)
        self.proj_radius = 15
        self.kp_initialized_empty = True
        self.use_molmo_vit_features = use_molmo_vit_features
        self.molmo_vit_feature_dim = molmo_vit_feature_dim
        self.molmo_vit_feature_scale = molmo_vit_feature_scale
        self.molmo_vit_feature_file = molmo_vit_feature_file
        self.molmo_vit_feature_expname = molmo_vit_feature_expname
        self.molmo_2d_expname = molmo_2d_expname
        self.use_pointnext_surface_features = use_pointnext_surface_features
        self.pointnext_surface_feature_dir = Path(pointnext_surface_feature_dir) if pointnext_surface_feature_dir else None
        self.pointnext_feature_dim = pointnext_feature_dim
        self.pointnext_feature_scale = pointnext_feature_scale
        self.pointnext_voxel_size = pointnext_voxel_size
        self.pointnext_max_disk_diameter = pointnext_max_disk_diameter
        self.pointnext_max_disk_depth_range = pointnext_max_disk_depth_range
        self.pointnext_max_disk_depth_ratio = pointnext_max_disk_depth_ratio
        self.pointnext_min_disk_points = pointnext_min_disk_points
        self.pointnext_max_surface_dist = pointnext_max_surface_dist
        hires = (np.asarray(self.res) * self.scale).astype(np.int64)
        super(KPNetGenerator, self).__init__(self.device, res=hires.tolist())

    @torch.inference_mode()
    def backproject_kps(self, mesh: Any, fragments: Any, cameras: CamerasBase, T: Any, kps: Any,
                        molmo_vit_features: torch.Tensor | None = None) -> Pointclouds:
        """Project 2D keypoint detections from multiple views into 3D space.

        For each view that has detected keypoints, draws colored circles at the
        detected locations onto a per-view color buffer. Each detected point gets
        a distinct color from COLOR_NAMES, which is then mapped back to a class ID
        via COLOR_MAP. The color buffer encodes per-pixel metadata as 3 uint8 channels.

        The depth buffer (fragments.zbuf) and camera parameters are used to unproject
        masked pixels into 3D world coordinates. A direction check filters out points
        whose backprojected ray faces away from the camera.

        The output Pointclouds carries a 4-channel uint8 feature vector per point:
          channel 0: view index — which rendered view this point came from
          channel 1: class ID — color-mapped point index within the view (from COLOR_MAP)
          channel 2: alpha — confidence/intensity from the drawn circle
          channel 3: valid — 1 if the backprojected ray faces the camera, 0 otherwise

        Args:
            mesh: PyTorch3D Meshes (unused here, kept for interface compatibility)
            fragments: Rasterization fragments containing zbuf depth [B, H, W, K]
            cameras: FoVPerspectiveCameras used for unprojection
            T: Lights (unused here, kept for interface compatibility)
            kps: Dict mapping view_index -> detected point coordinates from detect_kps

        Returns:
            Pointclouds with 3D positions and uint8 features [N, 4]
        """
        depth = fragments.zbuf[..., 0]  # [B, H, W], nearest face depth per pixel
        batch_size, imh, imw = depth.shape
        # Color buffer: 3 uint8 channels per pixel, initialized to zero (no detection)
        # Layout: [view_idx, class_id, alpha] per pixel
        color = torch.zeros((batch_size, 3, imh, imw), dtype=torch.uint8, device=self.device)

        for idx, kp_item in kps.items():
            # Draw colored circles at detected points. Each point gets a unique color
            # from COLOR_NAMES so we can recover which point is which after rasterization.
            cur_color = Image.new(mode='RGB', size=(imw, imh))
            Molmo.draw_points(cur_color, kp_item, radius=self.proj_radius, width=None, colors=cast(list[str | None], list(self.COLOR_NAMES.values())))
            cur_color = np.asarray(cur_color)  # [H, W, 3]
            cur_mask = np.any(cur_color, axis=-1)  # [H, W], True where a circle was drawn
            # Map drawn RGB colors back to integer class IDs via COLOR_MAP lookup
            cur_idx = [self.COLOR_MAP.get(a.tobytes(), 0) for a in cur_color[cur_mask, :3]]
            alpha = torch.from_numpy(cur_color[cur_mask, -1])
            cur_mask = torch.from_numpy(cur_mask)
            # Encode: channel 0 = view index, channel 1 = class ID, channel 2 = alpha
            color[idx, 0, cur_mask] = idx
            color[idx, 1, cur_mask] = torch.tensor(cur_idx, dtype=torch.uint8, device=self.device)
            color[idx, 2, cur_mask] = alpha.to(device=self.device, dtype=torch.uint8)

        # Mask: pixels with nonzero alpha (i.e., pixels covered by a drawn circle)
        mask = color[:, -1].bool()
        # Unproject masked pixels to 3D using depth and camera params
        pts_3d = get_depth_point_cloud(cameras, depth.unsqueeze(1), mask.unsqueeze(1))
        points, directions, origins = pts_3d.points_packed(), pts_3d.normals_packed(), pts_3d.features_packed()
        assert points is not None and directions is not None and origins is not None
        # Direction check: keep only points where the ray from origin through point
        # faces toward the camera (dot product > 0 means ray faces camera)
        valid_mask = torch.bmm((points - origins).unsqueeze(-2), directions.unsqueeze(-1)).view(-1) > 0

        if self.vis:
            pts_3d_ref = get_rgbd_point_cloud(cameras, color, depth.unsqueeze(1), mask.unsqueeze(1))
            assert torch.allclose(pts_3d_ref.points_packed(), points[valid_mask])

        # Concatenate the 3-channel color with the valid mask as 4th channel.
        # color.permute: [B, 3, H, W] -> [B, H, W, 3], then [mask] flattens to [N, 3].
        # Result without ViT cache: [N, 4] uint8 features = [view_idx, class_id, alpha, valid].
        point_features = torch.cat([color.permute(0, 2, 3, 1)[mask],
                                    valid_mask.to(dtype=color.dtype, device=color.device).unsqueeze(-1)], dim=-1)
        if molmo_vit_features is not None:
            mask_coords = mask.nonzero(as_tuple=False)  # [N, 3] in view, y, x order
            view_indices = mask_coords[:, 0].to(device=molmo_vit_features.device)
            sample_coords = mask_coords[:, [2, 1]].to(device=molmo_vit_features.device, dtype=torch.float32)
            sampled_features = self.sample_molmo_vit_features(molmo_vit_features, view_indices, sample_coords, imh, imw)
            point_features = torch.cat(
                [point_features.to(dtype=sampled_features.dtype, device=sampled_features.device), sampled_features],
                dim=-1)
        pts_3d = Pointclouds(
            points=points[np.newaxis],
            normals=directions[np.newaxis],
            features=point_features[np.newaxis])

        return pts_3d

    @staticmethod
    def sample2dfeat_bilinear(feat_2d: torch.Tensor, sample_coords: torch.Tensor, imh: int, imw: int) -> torch.Tensor:
        """
        Sample 2D feature maps with bilinear interpolation.

        Args:
            feat_2d: Tensor [B, C, H, W].
            sample_coords: Tensor [B, N, 2] in source-image pixel coordinates, order (x, y).
            imh: Source image height used by the coordinates.
            imw: Source image width used by the coordinates.

        Returns:
            Tensor [B, N, C].
        """
        bsz, _channels, _height, _width = feat_2d.shape
        num_samples = sample_coords.shape[1]
        coords_normed = sample_coords.clone().float()
        # Treat sample_coords as pixel-center coordinates in the source render/depth
        # image. With align_corners=False, normalized coordinates map feature-cell
        # centers rather than forcing the outermost input pixels onto the outermost
        # feature centers, which avoids an edge-dependent half-patch drift.
        coords_normed[..., 0] = (coords_normed[..., 0] + 0.5) * 2.0 / imw - 1.0
        coords_normed[..., 1] = (coords_normed[..., 1] + 0.5) * 2.0 / imh - 1.0
        grid = coords_normed.view(bsz, num_samples, 1, 2)
        sample_feats = F.grid_sample(
            feat_2d.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sample_feats.squeeze(-1).permute(0, 2, 1)

    def sample_molmo_vit_features(self, feat_maps: torch.Tensor, view_indices: torch.Tensor,
                                  sample_coords: torch.Tensor, imh: int, imw: int) -> torch.Tensor:
        """Sample reduced Molmo ViT feature maps at raw detection pixels."""
        if sample_coords.numel() == 0:
            return feat_maps.new_zeros((0, feat_maps.shape[-1]))
        feat_nchw = feat_maps.permute(0, 3, 1, 2).contiguous()
        sampled = feat_maps.new_zeros((sample_coords.shape[0], feat_maps.shape[-1]), dtype=torch.float32)
        for view_idx in view_indices.unique(sorted=True):
            point_mask = view_indices == view_idx
            view = int(view_idx.item())
            if view < 0 or view >= feat_nchw.shape[0]:
                continue
            coords = sample_coords[point_mask].view(1, -1, 2)
            sampled[point_mask] = self.sample2dfeat_bilinear(feat_nchw[view:view + 1], coords, imh, imw)[0]
        return sampled

    def _molmo_vit_projection(self, in_dim: int, out_dim: int, device: torch.device) -> torch.Tensor:
        """Create a deterministic Gaussian random projection matrix."""
        generator = torch.Generator(device='cpu')
        generator.manual_seed(0)
        projection = torch.randn(in_dim, out_dim, generator=generator, dtype=torch.float32) / math.sqrt(out_dim)
        return projection.to(device=device)

    def reduce_molmo_vit_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reduce cached Molmo ViT maps to the configured clustering dimension."""
        if features.ndim != 4:
            raise ValueError(f'Expected Molmo ViT features [V,H,W,C], got {tuple(features.shape)}')
        if features.shape[-1] == self.molmo_vit_feature_dim:
            return features.to(device=self.device, dtype=torch.float32)
        flat = features.to(device=self.device, dtype=torch.float32).reshape(-1, features.shape[-1])
        projection = self._molmo_vit_projection(features.shape[-1], self.molmo_vit_feature_dim, self.device)
        reduced = flat @ projection
        return reduced.reshape(*features.shape[:-1], self.molmo_vit_feature_dim)

    def reduce_pointnext_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project PointNeXt surface features to a compact clustering dimension."""
        if features.ndim != 2:
            raise ValueError(f'Expected PointNeXt features [4096,D], got {tuple(features.shape)}')
        features = features.to(device=self.device, dtype=torch.float32)
        if features.shape[-1] == self.pointnext_feature_dim:
            return features
        projection = self._molmo_vit_projection(features.shape[-1], self.pointnext_feature_dim, self.device)
        return features @ projection

    def _normal_disk_mask(self, pts: torch.Tensor, features: torch.Tensor, directions: torch.Tensor | None = None) -> torch.Tensor:
        """Keep only per-view disk backprojections without depth discontinuities.

        Molmo detections are drawn as 2D disks.  A normal view backprojects to a
        small nearly flat 3D patch; an abnormal view can span a depth
        discontinuity and produce candidates on disconnected front/back
        surfaces.  This filter operates independently for every drawn
        ``(view_idx, class_id)`` disk before snapping to the fixed surface anchors.

        The main rejection signal is depth spread along the local viewing
        direction.  A 2D disk that crosses a silhouette, chair-seat/leg boundary,
        or front/back surface edge has a much larger robust depth range than a
        normal disk on one continuous surface.  The older 3D diameter check is
        kept as a secondary guard against very large projected patches.
        """
        keep = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
        if pts.numel() == 0:
            return keep
        features_id = features[:, 0].long() * 256 + features[:, 1].long()
        for feature_id in features_id.unique(sorted=True):
            group_mask = features_id == feature_id
            group_pts = pts[group_mask]
            if group_pts.shape[0] < self.pointnext_min_disk_points:
                continue
            center = group_pts.median(dim=0).values
            lateral_diameter = None
            if directions is not None and (self.pointnext_max_disk_depth_range > 0 or self.pointnext_max_disk_depth_ratio > 0):
                group_dirs = directions[group_mask].to(device=pts.device, dtype=torch.float32)
                view_dir = group_dirs.mean(dim=0)
                view_dir_norm = torch.linalg.norm(view_dir)
                if view_dir_norm <= torch.finfo(torch.float32).eps:
                    continue
                view_dir = view_dir / view_dir_norm
                signed_depth = group_pts @ view_dir
                depth_q95 = torch.quantile(signed_depth, 0.95)
                depth_q05 = torch.quantile(signed_depth, 0.05)
                depth_range = depth_q95 - depth_q05
                if self.pointnext_max_disk_depth_range > 0 and depth_range > self.pointnext_max_disk_depth_range:
                    continue
                centered = group_pts - center
                depth_offsets = (centered @ view_dir)[:, None] * view_dir[None]
                lateral_radius = torch.quantile(torch.linalg.norm(centered - depth_offsets, dim=1), 0.95)
                lateral_diameter = 2.0 * lateral_radius
                if self.pointnext_max_disk_depth_ratio > 0:
                    ratio = depth_range / lateral_diameter.clamp_min(torch.finfo(torch.float32).eps)
                    if ratio > self.pointnext_max_disk_depth_ratio:
                        continue
            if self.pointnext_max_disk_diameter > 0:
                if lateral_diameter is None:
                    robust_radius = torch.quantile(torch.linalg.norm(group_pts - center, dim=1), 0.95)
                    robust_diameter = 2.0 * robust_radius
                else:
                    robust_diameter = lateral_diameter
                if robust_diameter > self.pointnext_max_disk_diameter:
                    continue
            keep[group_mask] = True
        return keep

    def _voxel_downsample_candidates(
        self,
        pts: torch.Tensor,
        features: torch.Tensor,
        pointnext_feat: torch.Tensor,
        surface_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxel-downsample snapped candidates without mixing view/color IDs."""
        if self.pointnext_voxel_size <= 0 or pts.shape[0] <= 1:
            return pts, features, pointnext_feat, surface_dist

        voxel = torch.floor(pts / self.pointnext_voxel_size).long()
        features_id = features[:, 0].long() * 256 + features[:, 1].long()
        keys = torch.cat([features_id[:, None], voxel], dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

        out_pts: list[torch.Tensor] = []
        out_features: list[torch.Tensor] = []
        out_pointnext_feat: list[torch.Tensor] = []
        out_surface_dist: list[torch.Tensor] = []
        raw_weights = features[:, 2].float().clamp_min(torch.finfo(torch.float32).eps)
        for idx in range(unique_keys.shape[0]):
            mask = inverse == idx
            weights = raw_weights[mask]
            weight_sum = weights.sum()
            weighted = weights[:, None]
            out_pts.append((pts[mask] * weighted).sum(dim=0) / weight_sum)
            out_pointnext_feat.append((pointnext_feat[mask] * weighted).sum(dim=0) / weight_sum)
            out_surface_dist.append((surface_dist[mask] * weights).sum(dim=0, keepdim=True) / weight_sum)
            feature_row = features[mask][weights.argmax()].float().clone()
            feature_row[2] = weight_sum.to(dtype=feature_row.dtype)
            feature_row[3] = 1
            out_features.append(feature_row)

        return (
            torch.stack(out_pts, dim=0),
            torch.stack(out_features, dim=0),
            torch.stack(out_pointnext_feat, dim=0),
            torch.cat(out_surface_dist, dim=0),
        )

    def snap_candidates_to_pointnext_surface(self, pts_3d: Pointclouds, surface_payload: dict[str, Any]) -> Pointclouds:
        """Snap valid raw candidates to saved PointNeXt anchors and attach features.

        The saved ``surface_xyz`` is the single source of truth for the 4096
        anchors.  Raw off-surface backprojections are only used to choose the
        nearest anchor; HDBSCAN then receives the snapped anchor coordinates plus
        a deterministic random projection of the corresponding PointNeXt feature.
        """
        all_pts = pts_3d.points_packed()
        all_features = pts_3d.features_packed()
        all_directions = pts_3d.normals_packed()
        if all_pts is None or all_features is None:
            return pts_3d

        valid_mask = all_features[:, 3].bool()
        raw_pts = all_pts[valid_mask].to(device=self.device, dtype=torch.float32)
        raw_features = all_features[valid_mask].to(device=self.device)
        raw_directions = all_directions[valid_mask].to(device=self.device, dtype=torch.float32) if all_directions is not None else None
        if raw_pts.shape[0] <= 1:
            return Pointclouds(points=raw_pts[None], features=raw_features[None])

        disk_mask = self._normal_disk_mask(raw_pts, raw_features, raw_directions)
        raw_pts = raw_pts[disk_mask]
        raw_features = raw_features[disk_mask]
        if raw_pts.shape[0] <= 1:
            return Pointclouds(points=raw_pts[None], features=raw_features[None])

        surface_xyz = surface_payload['surface_xyz'].to(device=self.device, dtype=torch.float32)
        surface_feat = surface_payload['surface_feat'].to(device=self.device, dtype=torch.float32)
        reduced_surface_feat = self.reduce_pointnext_features(surface_feat)

        dist = torch.cdist(raw_pts, surface_xyz)
        surface_nn_idx = dist.argmin(dim=1)
        surface_dist = dist[torch.arange(raw_pts.shape[0], device=raw_pts.device), surface_nn_idx]
        if self.pointnext_max_surface_dist is not None:
            surface_mask = surface_dist <= self.pointnext_max_surface_dist
            raw_features = raw_features[surface_mask]
            surface_nn_idx = surface_nn_idx[surface_mask]
            surface_dist = surface_dist[surface_mask]
            if surface_nn_idx.shape[0] <= 1:
                snapped = surface_xyz[surface_nn_idx]
                return Pointclouds(points=snapped[None], features=raw_features[None])

        snapped_xyz = surface_xyz[surface_nn_idx]
        candidate_feat = reduced_surface_feat[surface_nn_idx]
        snapped_xyz, raw_features, candidate_feat, surface_dist = self._voxel_downsample_candidates(
            snapped_xyz, raw_features, candidate_feat, surface_dist)

        # Keep the original metadata channels first.  HDBSCAN consumes the
        # feature tail as RandomProjection(PointNeXt feature); surface_dist is
        # computed above for optional filtering and diagnostics, but is not part
        # of the clustering space by default.
        point_features = torch.cat([raw_features.float(), candidate_feat], dim=1)
        return Pointclouds(points=snapped_xyz[None], features=point_features[None])

    def filter_draw_invalid_kps_with_images(self, images_with_kps: list[Image.Image], pts_3d: Pointclouds, kp: str, class_title: str, mesh_id: str) -> torch.Tensor:
        """Filter out invalid backprojections and annotate failed views with error text.

        Uses the uint8 feature vector to identify which (view, class) pairs have
        only invalid backprojections (valid=0) and no valid ones. For each such view,
        draws an error message on the corresponding image.

        Returns:
            valid_mask: Boolean tensor [N] indicating which points passed the
                        direction check in backproject_kps (features[:, 3] == 1).
        """
        # Features: [N, 4] uint8 = [view_idx, class_id, alpha, valid]
        features = pts_3d.features_packed()
        assert features is not None
        valid_mask = features[..., 3].bool()
        # Pack (view, class) IDs robustly for both original uint8 features and
        # extended float features with sampled Molmo ViT descriptors.
        valid_keypts = (features[valid_mask, 0].long() * 256 + features[valid_mask, 1].long()).unique().cpu().numpy()
        invalid_features = features[torch.logical_not(valid_mask)]
        invalid_keypts = (invalid_features[:, 0].long() * 256 + invalid_features[:, 1].long()).unique().cpu().numpy()
        # Find (view, class) pairs that are ONLY invalid (never valid in any pixel)
        setdiff = np.setdiff1d(invalid_keypts, valid_keypts, assume_unique=True)

        # Count failures per view and annotate the images
        for idx, count in Counter(int(x) // 256 for x in setdiff).items():
            ImageDraw.Draw(images_with_kps[idx]).text((10, 10), f"KPS back-projection failed {count} times", fill='red')

        print(f"Drawing {kp} in those images of {class_title}", flush=True)
        self.io.save_images(images_with_kps, class_title, mesh_id, postfix=kp, prefix='RawView')

        return valid_mask

    @torch.inference_mode()
    def aggregate_kps(self, mesh: Any, pts_3d: Pointclouds, max_points: int = 10000, kp_prompt: str = '', last_kp_cache: Any = None) -> Pointclouds:
        """Aggregate raw 3D point detections into consolidated keypoints via HDBSCAN.

        Points from backproject_kps carry a 4-channel uint8 feature vector per point:
          features[:, 0] - view index (uint8, which rendered view this point came from)
          features[:, 1] - point class ID (uint8, color-mapped index within the view)
          features[:, 2] - alpha/confidence weight (uint8)
          features[:, 3] - valid mask (uint8, 1 if backprojection ray faces camera)

        The first two channels are packed as ``view * 256 + class`` to compute
        per-class weight totals for cluster quality thresholding.
        """
        # Features are [view_idx, class_id, alpha, valid, optional_reduced_vit_features...].
        # The optional feature tail lets HDBSCAN cluster in [xyz, sampled_feature]
        # while centroid computation below still averages the original xyz points.
        features = pts_3d.features_packed()
        assert features is not None
        valid_mask = features[..., 3].bool()
        all_pts = pts_3d.points_packed()
        assert all_pts is not None
        pts = all_pts[valid_mask]  # [M, 3], keep only valid backprojections

        if pts.size(0) <= 1:
            return Pointclouds(pts[None])

        # Mean weight across all points (including invalid), used to normalize HDBSCAN weights.
        per_point_mean_weight = features[:, 2].float().mean(dtype=torch.float32)

        # Downsample if too many points, preserving metadata and optional sampled features.
        if pts.size(0) > max_points:
            pts_pc = Pointclouds(pts.unsqueeze(0), features=features[valid_mask].unsqueeze(0)).subsample(max_points)
            pts_feats = pts_pc.features_packed()
            pts_packed = pts_pc.points_packed()
            assert pts_feats is not None and pts_packed is not None
            pts = pts_packed
        else:
            pts_feats = features[valid_mask]

        raw_weights = pts_feats[:, 2].float()
        features_id = pts_feats[:, 0].long() * 256 + pts_feats[:, 1].long()
        sampled_vit_features = pts_feats[:, 4:].float() if pts_feats.shape[1] > 4 else None

        # Max total weight for any single semantic class — used as quality threshold
        per_point_max_weight = max(raw_weights[features_id == m].sum() for m in features_id.unique())
        # Normalize weights so HDBSCAN sees relative importance (typical max ~4)
        np_weights = (raw_weights / per_point_mean_weight.clamp_min(torch.finfo(torch.float32).eps)).cpu().numpy()
        np_pts = pts.cpu().numpy()
        if sampled_vit_features is not None and sampled_vit_features.numel() > 0:
            feat = sampled_vit_features
            feat = (feat - feat.mean(dim=0, keepdim=True)) / feat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            feature_scale = self.pointnext_feature_scale if self.use_pointnext_surface_features else self.molmo_vit_feature_scale
            np_pts = np.concatenate([np_pts, (feat * feature_scale).cpu().numpy()], axis=1)

        # Retry HDBSCAN with progressively smaller min_cluster_size until clusters are found
        min_cluster_size = 10
        clustered_masks: list[np.ndarray] = []
        while not clustered_masks and min_cluster_size >= 2:
            clustering = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
            cluster_assign = clustering.fit_predict(np_pts, sample_weight=np_weights)
            clustered_masks = [cluster_assign == i for i in np.unique(cluster_assign) if i != -1]
            min_cluster_size = math.ceil(min_cluster_size / 2)

        # Fallback: if no clusters found, treat all points as one cluster
        if not clustered_masks:
            clustered_masks = [np.ones(pts.size(0), dtype=bool)]

        # Build per-cluster point clouds with weights as features
        clustered_rawpts = Pointclouds(points=[pts[mask] for mask in clustered_masks],
                                       features=[raw_weights[mask].unsqueeze(-1) for mask in clustered_masks])

        # Helper to get features with type narrowing (features_packed() is never None for valid Pointclouds)
        def get_feats(p: Pointclouds) -> torch.Tensor:
            f = p.features_packed()
            assert f is not None
            return f

        # Compute weighted centroid for each cluster, keeping only clusters with
        # total weight > 2x the max single-class weight (ensures multi-view support)
        clustered_pts = Pointclouds([
            (p.points_packed() * get_feats(p)).sum(dim=0, keepdim=True) / total_w
            for p in clustered_rawpts
            if (total_w := get_feats(p).sum()) > 2 * per_point_max_weight
        ])
        # Fallback: if no cluster passes the threshold, keep all cluster centroids
        if not clustered_pts:
            clustered_pts = Pointclouds([
                (p.points_packed() * get_feats(p)).sum(dim=0, keepdim=True) / total_w
                for p in clustered_rawpts
                if (total_w := get_feats(p).sum())
            ])
        return clustered_pts

    def init_kps_batch(self, tensor_images: torch.Tensor, fragments: Any, kp: dict[Any, Any], class_title: str, mesh_id: str) -> dict[str, Any]:
        """Optional batch pre-processing before per-prompt detection.

        Base implementation returns an empty dict. Subclasses override this to
        extract patch features, pre-compute matches, or populate a cache keyed
        by prompt string that process_kp_list consults before calling detect_kps.

        Returns:
            Dict mapping prompt strings to pre-computed Pointclouds, or empty.
        """
        self.kp_initialized_empty = True
        cache: dict[str, Any] = {}
        if self.use_pointnext_surface_features:
            if self.pointnext_surface_feature_dir is None:
                print('[WARN] PointNeXt surface features requested without --pointnext-surface-feature-dir; clustering in raw xyz', file=sys.stderr)
            else:
                feature_path = surface_feature_path(self.pointnext_surface_feature_dir, class_title, mesh_id)
                if feature_path.exists():
                    try:
                        cache['__pointnext_surface__'] = load_surface_features(feature_path)
                    except ValueError as exc:
                        print(f'[WARN] Invalid PointNeXt surface feature artifact, clustering in raw xyz: {exc}', file=sys.stderr)
                else:
                    print(f'[WARN] PointNeXt surface feature cache not found, clustering in raw xyz: {feature_path}', file=sys.stderr)

        if not self.use_molmo_vit_features:
            return cache

        feature_expname = self.molmo_vit_feature_expname or self.expname
        feature_path = self.log_dir / feature_expname / class_title / mesh_id / self.molmo_vit_feature_file
        if not feature_path.exists():
            print(f'[WARN] Molmo ViT feature cache not found, clustering in xyz only: {feature_path}', file=sys.stderr)
            return cache

        payload = torch.load(feature_path, map_location='cpu')
        features = payload.get('features') if isinstance(payload, dict) else None
        if not isinstance(features, torch.Tensor):
            print(f'[WARN] Molmo ViT feature cache has no features tensor, clustering in xyz only: {feature_path}', file=sys.stderr)
            return cache

        reduced = self.reduce_molmo_vit_features(features)
        if reduced.shape[0] != tensor_images.shape[0]:
            print(
                f'[WARN] Molmo ViT feature view count {reduced.shape[0]} does not match rendered views {tensor_images.shape[0]}, '
                'clustering in xyz only',
                file=sys.stderr)
            return cache
        cache['__molmo_vit_features__'] = reduced
        return cache

    def load_cached_kps_2d(self, class_title: str, mesh_id: str, semantic_ids: list[Any], prompt: str) -> dict[int, Any] | None:
        """Load raw Molmo 2D detections from another experiment, if configured."""
        if self.molmo_2d_expname is None:
            return None
        semantic_token = ','.join(map(str, map(int, semantic_ids))) if semantic_ids else 'none'
        prompt_token = KPNetIO._safe_filename_token(prompt)
        cache_path = (
            self.log_dir / self.molmo_2d_expname / class_title / mesh_id /
            f'Molmo2D_{semantic_token}_{prompt_token}_kps2d.json'
        )
        if not cache_path.exists():
            print(f'[WARN] Cached Molmo 2D detections not found, re-querying Molmo: {cache_path}', file=sys.stderr)
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            print(f'[WARN] Failed to load cached Molmo 2D detections from {cache_path}: {exc}', file=sys.stderr)
            return None
        kps_2d = payload.get('kps_2d')
        if not isinstance(kps_2d, dict):
            print(f'[WARN] Cached Molmo 2D payload has no kps_2d dict, re-querying Molmo: {cache_path}', file=sys.stderr)
            return None
        return {int(view_idx): coords for view_idx, coords in kps_2d.items()}

    def process_kp_list(self, mesh: Meshes, fragments: Any, R: CamerasBase, T: Any, images: torch.Tensor, kp_list: dict[Any, Any], class_title: str, mesh_id: str, prompt_idx: int | slice = 0) -> dict[frozenset[Any], Pointclouds]:
        """Process all keypoint prompts: detect, backproject, and aggregate.

        This is the main orchestrator that ties the pipeline together for one mesh:
          1. init_kps_batch — optional batch pre-processing (e.g. patch extraction)
          2. For each unique prompt (e.g. "left eye"):
             a. detect_kps — query MLLM for 2D keypoints across all views
             b. backproject_kps — unproject 2D detections to 3D using depth + cameras,
                producing Pointclouds with uint8 features [view_idx, class_id, alpha, valid]
             c. aggregate_kps — cluster 3D points via HDBSCAN into final keypoints
          3. Cache results per prompt so duplicate prompts reuse previous detections.

        Args:
            mesh: PyTorch3D Meshes object
            fragments: Rasterization fragments (contains zbuf depth maps)
            R: Cameras (FoVPerspectiveCameras, named R for legacy reasons)
            T: Lights (PointLights, named T for legacy reasons)
            images: Rendered views [B, C, H, W]
            kp_list: Dict mapping semantic_id -> list of prompt strings.
                     kp_list[semantic_id][prompt_idx] is the prompt used for detection.
            class_title: Object class name (e.g. "airplane", "chair")
            mesh_id: Unique mesh identifier
            prompt_idx: Which prompt to use from each semantic_id's prompt list

        Returns:
            Dict mapping frozenset(semantic_ids) -> aggregated Pointclouds (one per prompt)
        """
        # Batch init: subclasses may extract features or pre-compute data here.
        # Returns a dict-like cache (e.g. SuperDict with npz_files and match_results).
        last_kp_cache = self.init_kps_batch(images, fragments, kp_list, class_title, mesh_id)

        # Group semantic IDs by their prompt string. Multiple semantic IDs can share
        # the same prompt (e.g. different annotations for "left wing tip").
        prompt_semantic_list = defaultdict(list)
        for semantic_id, kp in kp_list.items():
            prompt_semantic_list[kp[prompt_idx]].append(semantic_id)

        all_kps = {}
        for kp_prompt, semantic_ids in prompt_semantic_list.items():
            # Check cache: if init_kps_batch already produced results for this prompt,
            # reuse them. Otherwise, run MLLM detection.
            images_with_kps: List[Image.Image] = []
            if (kps_3d := last_kp_cache.get(kp_prompt, None)) is None:
                kps = self.load_cached_kps_2d(class_title, mesh_id, semantic_ids, kp_prompt)
                if kps is None:
                    # Detect 2D keypoints in all views for this prompt
                    kps, images_with_kps = self.detect_kps(images, kp_prompt, cat=class_title)
            else:
                # Cache hit: skip detection if not kp_initialized_empty
                kps = None if self.kp_initialized_empty else kps_3d

            if kps is not None:
                self.io.save_kps_2d_json(
                    class_title=class_title,
                    mesh_id=mesh_id,
                    semantic_ids=tuple(map(int, semantic_ids)),
                    prompt=kp_prompt,
                    kps_2d=kps,
                    num_views=int(images.size(0)),
                )
                # Backproject 2D detections -> 3D points with uint8 metadata,
                # optionally appending sampled Molmo ViT descriptors per raw point.
                kps_3d = self.backproject_kps(
                    mesh, fragments, R, T, kps,
                    molmo_vit_features=last_kp_cache.get('__molmo_vit_features__'))
                if self.vis and images_with_kps:
                    # Filter invalid backprojections and save debug visualizations
                    valid_mask = self.filter_draw_invalid_kps_with_images(images_with_kps, kps_3d, kp_prompt, class_title, mesh_id)
                    kps_pts = kps_3d.points_packed()
                    kps_feats = kps_3d.features_packed()
                    assert kps_pts is not None and kps_feats is not None
                    kps_3d_raw = Pointclouds(
                        points=kps_pts[np.newaxis, valid_mask],
                        features=kps_feats[np.newaxis, valid_mask, :3].to(dtype=torch.uint8))
                    self.io.save_kps(mesh, kps_3d_raw, class_title, mesh_id,
                                     semantic_id=','.join(map(str, semantic_ids)),
                                     postfix=kp_prompt, prefix='RawPts')
                if self.use_pointnext_surface_features and '__pointnext_surface__' in last_kp_cache:
                    kps_3d = self.snap_candidates_to_pointnext_surface(kps_3d, last_kp_cache['__pointnext_surface__'])
                # Cluster raw 3D points into consolidated keypoints
                kps_3d = self.aggregate_kps(mesh, kps_3d, kp_prompt=kp_prompt, last_kp_cache=last_kp_cache)
                if self.vis:
                    self.io.save_kps(mesh, kps_3d, class_title, mesh_id,
                                     semantic_id=','.join(map(str, semantic_ids)),
                                     postfix=kp_prompt, prefix='Pts')
                # Cache aggregated result for potential reuse by later prompts
                if self.kp_initialized_empty:
                    last_kp_cache[kp_prompt] = kps_3d.cpu()
            else:
                # Cache hit path: kps_3d was already set from the cache
                assert kps_3d is not None

            all_kps[frozenset(semantic_ids)] = kps_3d

        return all_kps

    def main_loop(
            self,
            use_texture: bool = False,
            save_rendered_images: bool = debug_enabled(),
            batch_size: int = 13,
            num_shards: int = 1,
            shard_id: int = 0,
            max_meshes: int | None = None) -> None:
        """Run the full detection pipeline over all test meshes.

        For each mesh: render views, detect keypoints via MLLM, backproject to
        3D, cluster, and save results. Skips meshes already processed (via
        check_if_complete). Catches IOError per mesh to continue on failures.

        Args:
            use_texture: Whether to render meshes with texture.
            save_rendered_images: Save multi-view renders as grid images.
            batch_size: Number of views to render per batch (memory trade-off).
            num_shards: Total number of shards for class-wise splitting.
            shard_id: 0-based shard index to run in this process.
            max_meshes: Optional cap of selected meshes to process.
        """
        if num_shards < 1:
            raise ValueError(f'num_shards must be >= 1, got {num_shards}')
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f'shard_id must be in [0, {num_shards}), got {shard_id}')

        skipped_meshes = self._load_skip_meshes()
        per_class_idx: dict[str, int] = {}
        selected_meshes = 0
        for mesh, keypoints, class_title, mesh_id, _pcd in self.io.loop_over_test_datasets(use_texture):
            if (class_title, mesh_id) in skipped_meshes:
                print(f'Skipping blacklisted mesh class {class_title} mesh {mesh_id}')
                continue
            class_idx = per_class_idx.get(class_title, 0)
            per_class_idx[class_title] = class_idx + 1
            if class_idx % num_shards != shard_id:
                continue
            if max_meshes is not None and selected_meshes >= max_meshes:
                break
            selected_meshes += 1
            try:
                kp_list = self.io.get_kp_names_from_lable(class_title, mesh_id, keypoints)
                # if mesh_id != 'e4e98f8654d29536dc858dada15498d2':
                #     continue
                if self.io.check_if_complete(kp_list, class_title, mesh_id):
                    print(f'Skipping class {class_title} mesh {mesh_id}')
                    continue

                print(f'Rendering class {class_title} mesh {mesh_id}')
                images, fragments, R, T = self.views_from_model(mesh, self.views, batch_size=batch_size, device=self.device)
                # images: [B, H, W, 4] or [B, H, W, 3] depending on renderer
                if self.scale != 1:
                    images = F.interpolate(images, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)

                # Convert to values between 0 and 255
                if images.dtype != torch.uint8 and images.ndim == 4:
                    images = (images * 255).clamp_(0, 255).to(torch.uint8)

                if save_rendered_images:
                    self.io.save_images(images, class_title, mesh_id)

                # kp_list = self.get_kp_names(images)
                all_kps = self.process_kp_list(mesh, fragments, R, T, images, kp_list, class_title, mesh_id)
                self.io.save_kps_with_semantic_ids(mesh, all_kps, class_title, mesh_id)
            except (IOError, torch.AcceleratorError, RuntimeError) as e:
                emsg = str(e)
                if 'illegal memory access' in emsg or 'cudaErrorIllegalAddress' in emsg:
                    self._record_skip_mesh(class_title, mesh_id, emsg)
                    skipped_meshes.add((class_title, mesh_id))
                    print(
                        f'[WARN] Added mesh to skip list due to CUDA illegal address: '
                        f'class={class_title} mesh={mesh_id}',
                        file=sys.stderr)
                print(f'Error with {e}', file=sys.stderr)

    def _skip_mesh_log_path(self) -> Path:
        """Path to persistent skip-list file shared by future eval runs."""
        return self.log_dir / self.expname / 'skipped_meshes.txt'

    def _load_skip_meshes(self) -> set[tuple[str, str]]:
        """Load persistent skip list of problematic meshes.

        File format (tab-separated):
            class_title<TAB>mesh_id<TAB>timestamp<TAB>reason
        """
        skip_file = self._skip_mesh_log_path()
        if not skip_file.exists():
            return set()
        skipped: set[tuple[str, str]] = set()
        with skip_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    skipped.add((parts[0], parts[1]))
        return skipped

    def _record_skip_mesh(self, class_title: str, mesh_id: str, reason: str) -> None:
        """Append a problematic mesh to persistent skip list with metadata."""
        skip_file = self._skip_mesh_log_path()
        skip_file.parent.mkdir(parents=True, exist_ok=True)
        reason_clean = ' '.join(reason.split())[:500]
        ts = datetime.now(timezone.utc).isoformat()
        with skip_file.open('a', encoding='utf-8') as f:
            f.write(f'{class_title}\t{mesh_id}\t{ts}\t{reason_clean}\n')
