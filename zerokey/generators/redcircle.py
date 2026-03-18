"""Red circle baseline generator for keypoint detection via CLIP-guided circle placement."""

from typing import Any

from PIL import Image
from einops import rearrange
import torch
from tqdm import trange

from zerokey.models.red_circle import RedCircle
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO


class RedCircleGenerator(KPNetGenerator[KPNetIO, RedCircle]):
    """Baseline that places red circles on images and scores them with CLIP similarity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vis = False

    def detect_kps(self, tensor_images: torch.Tensor, kp: str, cat: str = '') -> tuple[dict[int, Any], list[Image.Image]]:
        """Detect keypoints by finding the best red circle position per view.

        Returns:
            Tuple of per-view keypoint positions and visualization images.
        """
        all_kps: dict[int, Any] = {}
        all_vis: list[Image.Image] = []
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            all_vis.append(image)
            mask = tensor_images[idx, -1].bool()
            # Find the best circle position
            best_position, _best_similarity = self.multimodal.find_best_circle_position(
                image, kp, num_samples=100, mask=mask
            )
            all_kps[idx] = best_position
            if self.vis and best_position is not None:
                print(f"Drawing {kp} in this image", flush=True)
                self.multimodal.draw_points(image, best_position)
                image.show()
        return all_kps, all_vis
