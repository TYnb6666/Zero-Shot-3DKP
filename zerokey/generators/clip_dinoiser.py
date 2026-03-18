"""CLIP-DINOiser baseline generator using joint CLIP-DINO features for keypoint localization."""

from typing import Any

import torch
from PIL import Image
from einops import rearrange

from clip_dinoiser.pointmodel import ClipDINOiser
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO


class ClipDINOiserGenerator(KPNetGenerator[KPNetIO, ClipDINOiser]):
    """Baseline that localizes keypoints by matching CLIP text prompts against CLIP-DINO feature maps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vis = False

    def init_kps_batch(self, tensor_images: torch.Tensor, fragments: Any, kp: dict[Any, Any], class_title: str, mesh_id: str = '') -> dict[str, dict[int, Any]]:
        """Match all keypoint prompts against CLIP-DINO features across views in a single batch.

        Returns:
            Mapping from prompt name to per-view activation masks.
        """
        prompts = list(set(l[0] for l in kp.values()))
        models = self.multimodal.load_model_against_prompts(prompts)
        images = [Image.fromarray(rearrange(img, 'c h w -> h w c').cpu().numpy()).convert('RGB') for img in
                  tensor_images]
        masks = tensor_images[:, -1].bool()
        # Find the best circle position
        best_position = self.multimodal.visualize_per_image(images, masks, models)
        all_kps = {k: {i: best_position[i, j] for i in range(len(tensor_images)) if torch.any(best_position[i, j])} for
                   j, k in enumerate(prompts)}
        if self.vis:
            for k, kps in all_kps.items():
                for idx, mask in kps.items():
                    image = images[idx].copy()
                    self.multimodal.draw_points(image, mask)
                    image.save(self.log_dir / f'{class_title}_{k}_{idx}.png')
        return all_kps
