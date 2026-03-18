"""Red-circle saliency detector: find and mark the most CLIP-relevant region.

Uses a ViT-based saliency extractor to propose candidate positions, then
scores each candidate by drawing a red circle and measuring CLIP similarity
to a text query.  The highest-scoring position is returned.
"""

from __future__ import annotations

from typing import Sequence, Tuple, cast

import torch
from torch.nn import functional as F

from einops import rearrange
from PIL import Image, ImageDraw
from matplotlib import colors as mcolors

from feature_backprojection.saliency_extractor import ViTExtractor


class RedCircle:
    """CLIP-guided red-circle keypoint detector.

    Loads a ViT saliency extractor and a CLIP ViT-B/32 model on
    construction.  The main workflow is:

    1. Extract a saliency map from the image (or sample randomly).
    2. For each top-*k* salient pixel, draw a red circle and compute
       the CLIP similarity to a text query.
    3. Return the position with maximum similarity.
    """

    def __init__(self) -> None:
        import clip  # type: ignore[import-untyped]

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.vit_extractor = ViTExtractor(device=self.device)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Load an image from *image_path* and convert to RGB."""
        image = Image.open(image_path).convert('RGB')
        return image

    @staticmethod
    def estimate_radius(image: Image.Image) -> int:
        """Estimate a reasonable circle radius as 1/40 of the shorter side."""
        return min(image.width, image.height) // 40

    def compute_clip_similarity(self, image: Image.Image, text_query: str) -> float:
        """Return the cosine similarity between *image* and *text_query* under CLIP."""
        import clip  # type: ignore[import-untyped]

        image_input: torch.Tensor = cast(torch.Tensor, self.preprocess(image)).unsqueeze(0).to(self.device)
        text_input: torch.Tensor = clip.tokenize([text_query]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity: float = (image_features @ text_features.T).item()
        return similarity

    def extract_saliency_maps(
        self,
        image: Image.Image,
        load_size: int = 224,
        num_samples: int = 100,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract a ViT saliency map and return the top-*k* positions and values.

        Args:
            image: Input PIL image.
            load_size: Resolution passed to the ViT preprocessor.
            num_samples: Number of top-scoring pixels to return.
            mask: Optional ``(1, 1, H, W)`` mask to apply before selecting.

        Returns:
            ``(positions, values)`` where *positions* is ``(K, 2)`` integer
            ``[y, x]`` indices and *values* the corresponding saliency scores.
        """
        batch, _ = self.vit_extractor.preprocess(image, load_size=load_size)  # type: ignore[arg-type]
        batch = self.vit_extractor.extract_saliency_maps(batch.to(self.device))

        num_patches = self.vit_extractor.num_patches
        assert num_patches is not None
        h, w = num_patches
        batch = rearrange(batch, '1 (h w) -> 1 1 h w', h=h, w=w)
        batch = F.interpolate(batch, size=(image.height, image.width), mode="bicubic", align_corners=False)
        if mask is not None:
            batch *= mask

        top_k = torch.topk(batch.view(-1), k=num_samples, largest=True)
        selection = torch.zeros_like(batch, dtype=batch.dtype).view(-1)
        selection[top_k.indices] = top_k.values

        selection = rearrange(selection, '(h w) -> h w', h=image.height, w=image.width)
        return selection.nonzero(), selection[selection.nonzero(as_tuple=True)]

    def randon_sample_mask(
        self,
        image: Image.Image,
        num_samples: int = 100,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample *num_samples* random positions, optionally weighted by *mask*.

        Returns:
            ``(K, 2)`` integer tensor of ``[y, x]`` indices.
        """
        assert mask is not None
        batch = torch.rand(image.height, image.width, device=mask.device)
        batch *= mask

        top_k = torch.topk(batch.view(-1), k=num_samples, largest=True)
        selection = torch.zeros_like(batch, dtype=batch.dtype).view(-1)
        selection[top_k.indices] = top_k.values

        selection = rearrange(selection, '(h w) -> h w', h=image.height, w=image.width)
        return selection.nonzero()

    def find_best_circle_position(
        self,
        image: Image.Image,
        text_query: str,
        num_samples: int = 100,
        radius: int | None = None,
        mask: torch.Tensor | None = None,
    ) -> Tuple[Tuple[float, float] | None, float]:
        """Find the circle position that maximises CLIP similarity with *text_query*.

        Returns:
            ``(best_position, best_similarity)`` where *best_position* is
            normalised ``(x/w, y/h)`` or ``None`` if no samples were found.
        """
        width, height = image.size
        if radius is None:
            radius = self.estimate_radius(image)
        best_similarity: float = -1
        best_position: Tuple[float, float] | None = None
        samples = self.randon_sample_mask(image, num_samples=num_samples, mask=mask)
        samples = samples.long().cpu()

        for y, x in samples.numpy().tolist():
            image_with_circle = self.draw_circle_on_image(image, x, y, radius=radius)
            similarity = self.compute_clip_similarity(image_with_circle, text_query)
            if similarity > best_similarity:
                best_similarity = similarity
                best_position = (x / width, y / height)

        return best_position, best_similarity

    def draw_circle_on_image(
        self,
        image: Image.Image,
        x: float,
        y: float,
        radius: int | None = None,
        color: str = 'red',
        width: int | None = None,
        copy: bool = True,
    ) -> Image.Image:
        """Draw a circle on *image* at ``(x, y)`` and return the result.

        When *copy* is ``True`` (default) the original image is not modified.
        """
        if radius is None:
            radius = self.estimate_radius(image)
        image_with_circle = image.copy() if copy else image
        draw = ImageDraw.Draw(image_with_circle)
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=max(width or radius // 5, 5))
        return image_with_circle

    def draw_points(
        self,
        image: Image.Image,
        *kps: Tuple[float, float],
        radius: int = 5,
        width: int = 2,
        colors: Sequence[str] = tuple(mcolors.TABLEAU_COLORS.values()),  # type: ignore[assignment]
    ) -> None:
        """Draw circles at normalised ``(x, y)`` keypoint locations on *image* (in-place)."""
        w, h = image.width, image.height
        for kp, color in zip(kps, colors):
            x1 = kp[0] * w
            y1 = kp[1] * h
            color_str = str(color)

            if radius:
                # Draw circles at the specified points (eyes)
                self.draw_circle_on_image(image, x1, y1, radius=radius, color=color_str, width=width, copy=False)
            else:
                # Create an ImageDraw object
                draw = ImageDraw.Draw(image)
                draw.point((x1, y1), fill=color_str)
