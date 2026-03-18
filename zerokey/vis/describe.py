"""Point describability experiment: name and localize the most salient mesh vertex."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from pytorch3d.structures import Pointclouds
from matplotlib import pyplot as plt
from zerokey.rendering import views_from_model, debug_enabled
from kp_utils import sample_view_points
from zerokey.vis.base import VisGeneratorBase


class Generator(VisGeneratorBase):
    """Point describability pipeline: render a salient point, ask Molmo to name it, then re-detect.

    Selects the vertex with maximum saliency annotation, renders it as a red
    sphere, asks Molmo to name the marked part, then localizes that part name
    across views to measure describability.
    """

    def main_loop(self, vis: bool = True, batch_size: int = 1, debug: bool = debug_enabled()) -> None:
        """Run the describability pipeline over all test meshes.

        For each mesh: selects the most salient vertex, renders it as a red
        sphere, asks Molmo to name the marked part, then detects/backprojects/
        clusters that part across views.
        """
        views = sample_view_points(self.dist, 3)
        for mesh, class_title, mesh_id, annotations in self.io.loop_over_test_datasets():
            # Find the vertex with highest saliency score and use its 3D
            # position to render a red sphere marker across all views.
            selected_point = np.argmax(annotations)
            salience_score = annotations.max()
            print(f'Picking point {selected_point} with saliency {salience_score}')

            print(f'Rendering class {class_title} mesh {mesh_id}')

            # Give the selected point location to draw the red sphere
            images, fragments, R, T = views_from_model(self, mesh, views, sphere_location=mesh.verts_packed()[selected_point], batch_size=batch_size, device=self.device)
            images = F.interpolate(images, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)
            if vis:
                plt.imshow(rearrange(images[3], 'c h w -> h w c').clamp(min=0, max=1).cpu().numpy())
                plt.savefig('example.png')
            # Convert to values between 0 and 255
            images = images * 255
            images.clamp_(0, 255)
            images = images.to(torch.uint8)

            # Ask Molmo to name the red-dotted part, then use that name
            # as the detection prompt for re-localization across views.
            image = Image.fromarray(rearrange(images[0], 'c h w -> h w c').cpu().numpy())
            molmo_prompt = self.multimodal.generated_kps_points(image, text=f"Precisely name the part of the {class_title} marked by the red dot. Answer with just a name.")

            # Retrieve the prediction
            last_kp: dict[Any, Any] = {}
            for kp in [molmo_prompt]:
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
                    last_kp[kp] = kps_3d.cpu()
                self.io.save_kps(mesh, kps_3d, class_title, mesh_id, "schelling")

            print(last_kp)
