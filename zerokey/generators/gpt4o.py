"""Generator and model wrapper for GPT-4o-based keypoint localization."""

import base64
import io
import json
import sys
from typing import Any, Iterator

import torch
from einops import rearrange
from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import trange

from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO

from matplotlib import colors as mcolors


def pil_to_base64(pil_image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffered = io.BytesIO()
    # Save the image to the buffer in PNG format
    pil_image.save(buffered, format="PNG")
    # Get the byte data from the buffer
    img_data = buffered.getvalue()
    # Encode the byte data in base64
    base64_encoded = base64.b64encode(img_data).decode('utf-8')
    return base64_encoded


def norm_location_list(content: dict[str, Any], name: str | None = None) -> Iterator[tuple[Any, Any]]:
    """Recursively flatten nested GPT-4o location dicts into (name, coords) pairs.

    GPT-4o returns keypoint coordinates in varying nesting structures. This
    function normalizes them by recursing into dicts and enumerating lists,
    yielding ``(name, coords)`` tuples regardless of nesting depth.
    """
    for partname, location in content.items():
        if isinstance(location, list):
            yield from enumerate(location)
        elif isinstance(location, dict):
            yield from norm_location_list(location, partname)
        else:
            yield name, content
            break


class GPT4oLocalize:
    """Multimodal model wrapper that uses GPT-4o to localize keypoints in images.

    Sends images to the OpenAI API with JSON response format and parses
    the returned point coordinates.

    Attributes:
        image_wh: Width and height of the last queried image, used for
            coordinate normalization.
    """

    def __init__(self) -> None:
        self.image_wh: tuple[int, int] | None = None

    def get_kplist(self, client: OpenAI, image: Image.Image, kp: str) -> Any:
        """Query GPT-4o to locate a named keypoint in an image.

        Args:
            client: OpenAI API client.
            image: Input image to analyze.
            kp: Keypoint name to locate (e.g., 'left eye').

        Returns:
            OpenAI API response containing JSON with point coordinates.
        """
        self.image_wh = image.width, image.height
        image_encoded = pil_to_base64(image)
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_encoded}"}
                    },
                    {
                        "type": "text",
                        "text": f"Point to the {kp} in this image. Output the point locations in the image (xy) coordinate system"
                    }
                ]
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "json_object"
            }
        )
        return response

    def norm_location_list(self, content: dict[str, Any], name: str | None = None) -> Iterator[tuple[Any, Any]]:
        """Delegate to module-level norm_location_list."""
        return norm_location_list(content, name)

    def draw_points(self, image: Image.Image, kps: dict[str, Any], kps_wh: tuple[int, int] | None = None, radius: int = 5, width: int = 2, colors: Any = mcolors.TABLEAU_COLORS.values()) -> None:
        """Draw colored circles at detected keypoint locations on an image.

        Coordinates are normalized from the original query image dimensions
        to the target image dimensions.
        """
        # GPT-4o returns coordinates in the query image's pixel space (k_w x k_h).
        # Scale them to the target image's dimensions (w x h) for drawing.
        w, h = image.width, image.height
        wh = kps_wh if kps_wh is not None else self.image_wh
        assert wh is not None
        k_w, k_h = wh

        for kp, color in zip(kps.values(), colors):
            # GPT-4o returns coordinates in varying dict shapes: try 'x'/'y' keys
            # first, then fall back to iterating dict values, then top-level values.
            try:
                x1 = kp['x']
                y1 = kp['y']
            except Exception:
                try:
                    x1, y1 = kp.values()
                except Exception:
                    x1, y1 = kps.values()

            # Normalize from query image coords to target image coords
            x1 = float(x1) / k_w * w
            y1 = float(y1) / k_h * h

            # Create an ImageDraw object
            draw = ImageDraw.Draw(image)

            if radius:
                # Draw circles at the specified points (eyes)
                if width:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), outline=str(color), width=width)
                else:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=str(color))
            else:
                draw.point((x1, y1), fill=str(color))


class GPT4oGenerator(KPNetGenerator[KPNetIO, GPT4oLocalize]):
    """Baseline generator using GPT-4o for both keypoint naming and localization.

    Overrides detect_kps to query GPT-4o per view instead of using Molmo,
    parsing JSON responses for point coordinates.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vis = False

    def detect_kps(self, tensor_images: torch.Tensor, kp: str, cat: str = '') -> tuple[dict[int, Any], list[Image.Image]]:
        """Detect keypoints by querying GPT-4o for each rendered view.

        Returns:
            Tuple of (per-view keypoint dict, list of visualization images).
        """
        all_kps: dict[int, Any] = {}
        all_vis: list[Image.Image] = []
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            all_vis.append(image)
            # Find the best circle position
            response = self.multimodal.get_kplist(self.gpt.client, image, kp)
            try:
                assert response.choices[0].message.content is not None
                content = json.loads(response.choices[0].message.content)
                kps = {k: v for k, v in self.multimodal.norm_location_list(content)}
            except Exception as e:
                pbar.clear()
                print(f'{kp}: Paring {response} encountered {e}', file=sys.stderr)
                continue
            all_kps[idx] = kps
            if self.vis:
                print(f"Drawing {kp} in this image", flush=True)
                self.multimodal.draw_points(image, kps)
                image.show()
        return all_kps, all_vis
