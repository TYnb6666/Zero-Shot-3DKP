"""Schelling point experiment: detect keypoints via GPT-4o naming + Molmo localization."""

from contextlib import suppress, nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.structures import Pointclouds
from matplotlib import pyplot as plt
from zerokey.rendering import views_from_model, debug_enabled
from kp_utils import sample_view_points
from zerokey.vis.base import VisGeneratorBase


class Generator(VisGeneratorBase):
    """Schelling point keypoint detection using GPT-4o for naming and Molmo for localization.

    Renders multi-view images of 3D meshes, queries GPT-4o for keypoint names,
    then uses Molmo to localize each named keypoint across views.
    """

    def main_loop(self, vis: bool = False, batch_size: int = 1, debug: bool = debug_enabled()) -> None:
        """Run the full Schelling point pipeline over all test meshes.

        For each mesh: render views, get keypoint names from GPT-4o,
        detect/backproject/cluster keypoints, and save results.
        """
        views = sample_view_points(self.dist, 3)
        for mesh, class_title, mesh_id, _ in self.io.loop_over_test_datasets():
            with nullcontext() if True else suppress(Exception):

                print(f'Rendering class {class_title} mesh {mesh_id}')
                images, fragments, R, T = views_from_model(self, mesh, views, batch_size=batch_size, device=self.device)
                images = F.interpolate(images, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)
                if vis:
                    plt.imshow(rearrange(images[0], 'c h w -> h w c').clamp(min=0, max=1).cpu().numpy())
                    plt.show()
                # Convert to values between 0 and 255
                images = images * 255
                images.clamp_(0, 255)
                images = images.to(torch.uint8)
                kp_list = self.get_prompts(images)

                last_kp: dict[Any, Any] = {}
                for kp in kp_list:
                    if (kps_3d := last_kp.get(kp[-1], None)) is None:
                        kps, _ = self.detect_kps(images, kp[-1], class_title)
                        kps_3d = self.backproject_kps(mesh, fragments, R, T, kps)
                        kps_pts = kps_3d.points_packed()
                        kps_feat = kps_3d.features_packed()
                        assert kps_pts is not None and kps_feat is not None
                        kps_3d_raw = Pointclouds(points=kps_pts[None],
                                                 features=kps_feat[None, ..., :3].to(dtype=torch.uint8))
                        self.io.save_kps(mesh, kps_3d_raw, class_title, mesh_id, "schelling", postfix='rawpts')
                        kps_3d = self.aggregate_kps(mesh, kps_3d)
                        last_kp[kp[0]] = kps_3d.cpu()
                    self.io.save_kps(mesh, kps_3d, class_title, mesh_id, "schelling")
