"""DINO saliency baseline generator for unsupervised keypoint detection."""

from typing import Any

import torch
import torchvision
from PIL import Image
from einops import rearrange
from matplotlib import colors as mcolors
from tqdm import trange

from zerokey.models.red_circle import RedCircle
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO


class DINOSaliency(RedCircle):
    """Extends RedCircle with DINO-based saliency map extraction for keypoint localization."""
    def find_salient_position(self, image: Image.Image, num_samples: int = 100, mask: torch.Tensor | None = None) -> list[tuple[float, float]]:
        """Find the circle position that maximizes the CLIP similarity with the text query."""
        width, height = image.size
        best_position = []
        bbox_w = 0.01 * min(width, height)
        samples, scores = self.extract_saliency_maps(image, num_samples=num_samples, mask=mask)
        bboxes = torch.concat((samples - bbox_w, samples + bbox_w), dim=-1)
        bboxes = torchvision.ops.nms(boxes=bboxes, scores=scores, iou_threshold=0.5)
        samples = samples.long()

        for y, x in samples[bboxes].cpu().numpy().tolist():
            best_position.append((x / width, y / height))

        return best_position

    def draw_points(self, image: Image.Image, kps: Any, radius: int = 5, width: int = 2, colors: Any = mcolors.TABLEAU_COLORS.values()) -> None:
        """Unpack saliency point list and delegate to parent draw_points."""
        return super().draw_points(image, *kps, radius=radius, width=width, colors=colors)


class SaliencyGenerator(KPNetGenerator[KPNetIO, DINOSaliency]):
    """Baseline that detects keypoints using DINO saliency maps with NMS filtering."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vis = False

    @torch.inference_mode()
    def get_kp_names(self, tensor_images: torch.Tensor) -> list[str]:
        """Return a fixed keypoint name since saliency detection is class-agnostic."""
        return ['salient points']

    def detect_kps(self, tensor_images: torch.Tensor, kp: str, cat: str = '') -> tuple[dict[int, Any], list[Image.Image]]:
        """Detect salient positions in each view using DINO saliency maps.

        Returns:
            Tuple of per-view salient point lists and visualization images.
        """
        all_kps: dict[int, Any] = {}
        all_vis: list[Image.Image] = []
        pbar = trange(tensor_images.size(0))
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            all_vis.append(image)
            mask = tensor_images[idx, -1].bool()
            # Find the best circle position
            best_position = self.multimodal.find_salient_position(image, num_samples=100, mask=mask)
            all_kps[idx] = best_position
            if self.vis:
                print(f"Drawing salient position in this image", flush=True)
                self.multimodal.draw_points(image, best_position)
                image.show()
        return all_kps, all_vis
