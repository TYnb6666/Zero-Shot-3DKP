"""Zero-shot 3D keypoint detection pipeline using multi-view MLLM queries and HDBSCAN clustering."""

import math
import os
import sys
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
                 res: int | tuple[int, int] = 512, scale: int | float = 2):
        """Initialize the generator with output paths, rendering config, and viewpoints.

        Args:
            log_dir: Root output directory for experiment results.
            expname: Experiment name used as subdirectory under log_dir.
            res: Base render resolution (pixels). Actual rendering is at res*scale.
            scale: Upscale factor; renders at res*scale then downscales for anti-aliasing.
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
        hires = (np.asarray(self.res) * self.scale).astype(np.int64)
        super(KPNetGenerator, self).__init__(self.device, res=hires.tolist())

    @torch.inference_mode()
    def backproject_kps(self, mesh: Any, fragments: Any, cameras: CamerasBase, T: Any, kps: Any) -> Pointclouds:
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
            cast(Molmo, self.multimodal).draw_points(cur_color, kp_item, radius=self.proj_radius, width=None, colors=cast(list[str | None], list(self.COLOR_NAMES.values())))
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
        # Result: [N, 4] uint8 features = [view_idx, class_id, alpha, valid]
        color = torch.cat([color.permute(0, 2, 3, 1)[mask],
                           valid_mask.to(dtype=color.dtype, device=color.device).unsqueeze(-1)], dim=-1)
        pts_3d = Pointclouds(
            points=points[np.newaxis],
            normals=directions[np.newaxis],
            features=color[np.newaxis])

        return pts_3d

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
        valid_mask = features[..., -1].bool()
        # Reinterpret first 2 uint8 channels as int16 for packed (view, class) IDs
        valid_keypts = features[valid_mask, :2].view(dtype=torch.int16).unique().cpu().numpy()
        invalid_keypts = features[torch.logical_not(valid_mask), :2].view(dtype=torch.int16).unique().cpu().numpy()
        # Find (view, class) pairs that are ONLY invalid (never valid in any pixel)
        setdiff = np.setdiff1d(invalid_keypts, valid_keypts, assume_unique=True)

        # Count failures per view and annotate the images
        for idx, count in Counter(x.tobytes()[0] for x in setdiff).items():
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

        The first two uint8 channels are reinterpreted as a single int16 via
        .view(torch.int16) to form a packed (view, class) identifier per point.
        This is used to compute per-class weight totals for cluster quality thresholding.
        """
        # Features are uint8 RGBA: [view_idx, class_id, alpha, valid]
        features = pts_3d.features_packed()  # [N, 4], dtype=uint8
        assert features is not None
        features_id, raw_weights, valid_mask = features[:, :2], features[:, 2], features[..., -1].bool()
        all_pts = pts_3d.points_packed()
        assert all_pts is not None
        pts = all_pts[valid_mask]  # [M, 3], keep only valid backprojections

        if pts.size(0) <= 1:
            return Pointclouds(pts[None])

        # Mean weight across all points (including invalid), used to normalize HDBSCAN weights
        per_point_mean_weight = raw_weights.mean(dtype=torch.float32)

        # Downsample if too many points, preserving features for class ID recovery
        if pts.size(0) > max_points:
            pts_pc = Pointclouds(pts.unsqueeze(0), features=features[valid_mask].unsqueeze(0)).subsample(max_points)
            # Reinterpret 2 uint8 channels as one int16 to get packed (view, class) identifier
            pts_feats = pts_pc.features_packed()
            assert pts_feats is not None
            features_id = pts_feats[:, :2].view(torch.int16).view(-1)
            raw_weights = pts_feats[:, 2]  # [M]
            pts_packed = pts_pc.points_packed()
            assert pts_packed is not None
            pts = pts_packed
        else:
            raw_weights = raw_weights[valid_mask]
            # Reinterpret 2 uint8 channels as one int16 to get packed (view, class) identifier
            features_id = features_id[valid_mask].view(torch.int16).view(-1)

        # Max total weight for any single semantic class — used as quality threshold
        per_point_max_weight = max(raw_weights[features_id == m].sum() for m in features_id.unique())
        # Normalize weights so HDBSCAN sees relative importance (typical max ~4)
        np_weights = (raw_weights / per_point_mean_weight).cpu().numpy()
        np_pts = pts.cpu().numpy()

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
        return {}

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
                # Detect 2D keypoints in all views for this prompt
                kps, images_with_kps = self.detect_kps(images, kp_prompt, cat=class_title)
            else:
                # Cache hit: skip detection if not kp_initialized_empty
                kps = None if self.kp_initialized_empty else kps_3d

            if kps is not None:
                # Backproject 2D detections -> 3D points with uint8 features
                kps_3d = self.backproject_kps(mesh, fragments, R, T, kps)
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

    def main_loop(self, use_texture: bool = False, save_rendered_images: bool = debug_enabled(), batch_size: int = 13) -> None:
        """Run the full detection pipeline over all test meshes.

        For each mesh: render views, detect keypoints via MLLM, backproject to
        3D, cluster, and save results. Skips meshes already processed (via
        check_if_complete). Catches IOError per mesh to continue on failures.

        Args:
            use_texture: Whether to render meshes with texture.
            save_rendered_images: Save multi-view renders as grid images.
            batch_size: Number of views to render per batch (memory trade-off).
        """
        for mesh, keypoints, class_title, mesh_id, _pcd in self.io.loop_over_test_datasets(use_texture):
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
            except IOError as e:
                print(f'Error with {e}', file=sys.stderr)


