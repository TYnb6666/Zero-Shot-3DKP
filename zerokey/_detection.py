"""KeypointDetectionMixin: shared keypoint detection logic for generators and visualizations."""

import json
import sys
from collections import OrderedDict
from typing import Any, ClassVar, Dict, List, Tuple, cast
from xml.etree.ElementTree import ParseError

import torch
from PIL import Image, ImageColor, ImageDraw
from einops import rearrange
from pytorch3d.structures import Pointclouds
from tqdm import trange
from matplotlib import colors as mcolors
from matplotlib.typing import ColorType

from zerokey.models import GPT4o, Molmo


class KeypointDetectionMixin:
    """Mixin providing shared keypoint detection, prompting, and color-collapse logic.

    Requires the host class to provide:
        gpt: GPT4o instance
        vis: bool flag for debug visualization
        device: torch.device
        multimodal: MLLM model instance (e.g. Molmo)
    """

    COLOR_NAMES: ClassVar[OrderedDict[str, ColorType]] = OrderedDict(
        a for a in mcolors.CSS4_COLORS.items() if any(ImageColor.getrgb(a[1]))
    )
    # Reverse lookup: 3-byte RGB hex -> integer class ID.
    COLOR_MAP: ClassVar[Dict[bytes, int]] = {
        bytes.fromhex(str(s).lstrip('#')): v for v, s in enumerate(COLOR_NAMES.values())
    }

    # Type stubs for attributes/properties provided by the host class
    gpt: GPT4o  # cached_property in subclasses
    vis: bool
    device: torch.device
    multimodal: Any

    @torch.inference_mode()
    def get_prompts(self, tensor_images: torch.Tensor) -> List[str]:
        """Query GPT-4o to name salient keypoints visible in the first rendered view."""
        image = Image.fromarray(rearrange(tensor_images[0], 'c h w -> h w c').cpu().numpy())
        response = self.gpt.get_kplist(image)

        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message or not choice.message.content:
            print("Warning: No content received from GPT.", file=sys.stderr)
            return []

        content = json.loads(choice.message.content)
        kp_list = [kp for kp in self.gpt.iter_over_list(content)]
        print(','.join(kp_list), flush=True)
        return kp_list

    @torch.inference_mode()
    def detect_kps(self, tensor_images: torch.Tensor, kp: str, cat: str = '') -> Tuple[Dict[int, Any], List[Image.Image]]:
        """Detect 2D keypoints in rendered views using the MLLM (Molmo).

        For each view, prompts the MLLM to locate the keypoint described by `kp`.
        When `cat` is non-empty, includes the category in the prompt for context.

        Args:
            tensor_images: Rendered views [B, C, H, W], uint8
            kp: Keypoint prompt string (e.g. "left eye", "nose tip")
            cat: Object category name for prompt context (e.g. "airplane")

        Returns:
            all_kps: Dict mapping view_index -> detected point coordinates.
            all_vis: List of PIL Images for visualization/debugging.
        """
        molmo = cast(Molmo, self.multimodal)
        prompt_suffix = f"this {cat}" if cat else "this image"

        all_kps: Dict[int, Any] = {}
        all_vis: List[Image.Image] = []
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy())
            all_vis.append(image)
            kps = molmo.generated_kps_points(image, text=f"point to the {kp} on {prompt_suffix}")
            try:
                kps, alt = molmo.parse_points_str(kps)
            except ParseError as e:
                pbar.clear()
                print(f'{kp}: Paring {str.strip(kps)} encountered {e}', file=sys.stderr)
                if self.vis:
                    ImageDraw.Draw(image).text((10, 10), f"No kps for {str.strip(kps)}\nError: {e}", fill='red')
                continue
            if len(kps) >= 8:
                pbar.clear()
                print(f'{kp}: too many kps for {alt}: {len(kps)}', file=sys.stderr)
                if self.vis:
                    ImageDraw.Draw(image).text((10, 10), f"Too many kps for {alt}", fill='red')
                continue
            if self.vis:
                molmo.draw_points(image, kps)
            all_kps[idx] = kps
        return all_kps, all_vis

    def collapse_same_color(self, pts_3d: Pointclouds) -> Pointclouds:
        """Merge 3D points that share the same drawn color into single centroids."""
        pts, color = pts_3d.points_packed(), pts_3d.features_packed()
        assert pts is not None and color is not None
        color_value = map(ImageColor.getrgb, self.COLOR_NAMES.values())
        color_value = torch.as_tensor(list(color_value), dtype=torch.float32, device=self.device)
        cluster = torch.isclose(color_value[:, None, :], color[..., :3], atol=1e-2).all(dim=-1)
        idx, = cluster.any(dim=-1).nonzero(as_tuple=True)
        mini_cluster = torch.stack([pts[cluster[i]].mean(dim=0) for i in idx])
        mini_color = torch.stack([color[cluster[i]].mean(dim=0) for i in idx])
        return Pointclouds(mini_cluster, features=mini_color)
