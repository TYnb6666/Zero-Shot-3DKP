"""PaliGemma baseline generator using Google PaliGemma for keypoint detection and segmentation."""

from typing import Any

from PIL import Image, ImageDraw
from einops import rearrange
import torch
from tqdm import trange

from data_creation.pali_gemma import PaliGemma
from kp_utils.rendering import sample_view_points
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO

from pytorch3d.structures import Pointclouds


class PaliGemmaGenerator(KPNetGenerator[KPNetIO, PaliGemma]):
    """Baseline that uses PaliGemma detect/segment instructions for keypoint localization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vis = True
        self.views = sample_view_points(self.dist, partition=1)

    def init_kps_batch(self, tensor_images: torch.Tensor, fragments: Any, kp_list: dict[Any, Any], class_title: str, mesh_id: str) -> dict[str, Pointclouds]:
        """Load previously saved keypoint point clouds for each unique prompt.

        Returns:
            Mapping from prompt name to aggregated Pointclouds.
        """
        prompts = {k for _, kp in kp_list.items() for k in kp}
        all_kps = {v: Pointclouds(pts) for v in prompts if
                   (pts := list(
            self.io.load_kps_with_semantic_ids(class_title, mesh_id, prefix='Pts', postfix=v).values()
            ))}
        return all_kps
    
    def detect_kps(self, tensor_images: torch.Tensor, kp: str) -> tuple[dict[int, Any], list[Image.Image]]:
        """Detect keypoints per view using PaliGemma detect and segment instructions.

        Returns:
            Tuple of per-view keypoint positions and visualization images.
        """
        all_kps = {}
        all_vis = []
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            all_vis.append(image)
            try:  # Call the detect and segment instructions
                kps = self.multimodal.generated_kps_points(
                    image, text=f"detect {kp}"
                ), self.multimodal.generated_kps_points(
                    image, text=f"segment {kp}"
                )
            except Exception as e:
                print(f"Error segmenting {kp}: {e}")
                # Draw error text on image
                if self.vis:
                    ImageDraw.Draw(image).text((10, 10), f"No kps for {str.strip(kp)}\nError: {e}", fill='red')
                continue
            if self.vis:
                self.multimodal.draw_points(image, kps)
            all_kps[idx] = kps
        return all_kps, all_vis
