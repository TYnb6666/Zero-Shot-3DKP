"""Stable keypoints baseline using unsupervised keypoint regression from Stable Diffusion features."""

from typing import Any

import torch
from PIL import Image
from einops import rearrange
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds

from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO
from unsupervised_keypoints import MainKeypointRegressor


class StableKeypoints(KPNetGenerator[KPNetIO, MainKeypointRegressor]):
    """Baseline that detects keypoints via unsupervised Stable Diffusion feature regression."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vis = True

    def init_kps_batch(self, tensor_images: torch.Tensor, fragments: Any, kp: dict[Any, Any], class_title: str, mesh_id: str = '') -> dict[str, dict[int, Any]]:
        """Extract top-k keypoint embeddings for all views in a single batch.

        Returns:
            Mapping from semantic keypoint name to per-view detected point coordinates.
        """
        images = [Image.fromarray(rearrange(img, 'c h w -> h w c').cpu().numpy()) for img in
                  tensor_images]
        num_points = len(kp)
        points = self.multimodal.get_embeddings(images, top_k=num_points)
        all_kps = {k: {i: points[i, j] for i in range(len(tensor_images))} for j, k in enumerate(kp)}
        if self.vis:
            for k, kps in all_kps.items():
                for idx, pts in kps.items():
                    image = images[idx].copy()
                    self.multimodal.draw_points(image, pts)
                    image.save(self.log_dir / f'{class_title}_{k}_{idx}.png')
        return all_kps

    def process_kp_list(self, mesh: Meshes, fragments: Any, R: CamerasBase, T: Any, images: torch.Tensor, kp_list: dict[Any, Any], class_title: str, mesh_id: str, prompt_idx: int | slice = 0) -> dict[frozenset[Any], Pointclouds]:
        """Backproject and aggregate each semantic keypoint individually, saving raw and clustered results."""
        last_kp = self.init_kps_batch(images, fragments, kp_list, class_title, mesh_id)
        all_kps: dict[frozenset[Any], Pointclouds] = {}
        for semantic_id, kps in last_kp.items():
            kps_3d = self.backproject_kps(mesh, fragments, R, T, kps)
            points = kps_3d.points_packed()
            features = kps_3d.features_packed()
            assert points is not None and features is not None
            kps_3d_raw = Pointclouds(points=points[None],
                                     features=features[None, ..., :3].to(dtype=torch.uint8))
            self.io.save_kps(mesh, kps_3d_raw, class_title, mesh_id, semantic_id, postfix='rawpts')
            kps_3d = self.aggregate_kps(mesh, kps_3d)

            self.io.save_kps(mesh, kps_3d, class_title, mesh_id, semantic_id)
            all_kps[frozenset([semantic_id])] = kps_3d

            self.vis = False
        return all_kps
