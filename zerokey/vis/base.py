"""Base class for visualization generators with shared Molmo detection and backprojection."""

import os
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
from sklearn.cluster import HDBSCAN
from zerokey.rendering import RenderO3D
from zerokey.models import GPT4o, Molmo
from zerokey._detection import KeypointDetectionMixin
from zerokey.io.schelling import SchellingIO


class VisGeneratorBase(KeypointDetectionMixin, RenderO3D):
    """Shared base for Schelling and describability visualization generators.

    Provides GPT-4o naming, Molmo localization, RGBD backprojection, color
    collapsing, and HDBSCAN clustering. Subclasses implement ``main_loop()``.

    Attributes:
        gpt: GPT-4o instance for keypoint name generation.
        multimodal: Lazily-initialized Molmo instance for point localization.
        io: SchellingIO handler for dataset iteration and output saving.
    """

    Multimodal = Molmo

    def __init__(self, log_dir: str | os.PathLike[str] = Path(), expname: str = 'MolmoPTSNewPrompt') -> None:
        self.log_dir = Path(log_dir)
        self.io = SchellingIO(self.log_dir / expname)
        self.dist = 2.0
        self.res = 512
        self.scale = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vis = False
        super(VisGeneratorBase, self).__init__(self.device, res=self.scale * self.res)

    @cached_property
    def gpt(self) -> GPT4o:
        """Lazy-initializing GPT-4o instance (only needed for get_prompts)."""
        return GPT4o()

    @cached_property
    def multimodal(self) -> Molmo:
        """Lazy-initializing multimodal model instance."""
        return self.Multimodal()

    @torch.inference_mode()
    def backproject_kps(self, mesh: Any, fragments: Any, R: Any, T: Any, kps: dict[int, list[Any]]) -> Pointclouds:
        """Project 2D keypoint detections into 3D using depth buffers and camera parameters.

        Draws colored circles at detected locations, then uses RGBD unprojection
        to lift points into 3D space. Color encodes keypoint identity.

        Returns:
            Pointclouds with 4-channel features (RGB color + depth).
        """
        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
        depth = rearrange(fragments.zbuf[..., 0], 'N H W -> N 1 H W')
        mask = torch.zeros_like(depth, dtype=torch.bool)
        color = torch.zeros((depth.size(0), 4, depth.size(2), depth.size(3)), dtype=torch.float32, device=self.device)

        imh, imw = mask.shape[2:]
        # Assign a unique CSS4 color to each detected point across all views,
        # advancing through the color iterator so no two points share a color.
        color_iter = list(self.COLOR_NAMES.values())
        for idx, kp_item in kps.items():
            cur_color = Image.new(mode='RGB', size=(imw, imh))
            self.multimodal.draw_points(cur_color, kp_item, radius=0, colors=color_iter)  # type: ignore[arg-type]
            cur_color = rearrange(np.asarray(cur_color), 'H W C -> C H W')
            mask[idx, 0, np.any(cur_color > 0, axis=0)] = 1
            color[idx, :3] = torch.from_numpy(cur_color)

        color[:, 3] = depth[:, 0]
        pts_3d = get_rgbd_point_cloud(cameras, color, depth, mask)

        if self.vis:
            self.io.save_mesh(mesh, self.log_dir / "output_mesh.ply")
            pts_packed = pts_3d.points_packed()
            assert pts_packed is not None
            self.io.save_pointcloud(Pointclouds(
                pts_packed[None]), self.log_dir / "output_pointcloud.ply")
        return pts_3d

    @torch.inference_mode()
    def aggregate_kps(self, mesh: Any, pts_3d: Pointclouds, kp_prompt: str = '', last_kp_cache: Any = None) -> Pointclouds:
        """Cluster backprojected 3D points via HDBSCAN into final keypoint locations.

        Uses HDBSCAN with epsilon-based cluster selection to group nearby 3D
        points. Computes the mean position per cluster as the final keypoint.
        """
        pts = pts_3d.points_packed()
        assert pts is not None

        clustering = HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=0.1, allow_single_cluster=True)
        cluster_assign = clustering.fit_predict(pts.cpu().numpy())
        clustered_pts = Pointclouds(torch.stack(
            [pts[cluster_assign == i].mean(dim=0) for i in np.unique(cluster_assign)])[None])

        if self.vis:
            self.io.save_pointcloud(clustered_pts, self.log_dir / "output_pointcloud_c.ply")
        return clustered_pts
