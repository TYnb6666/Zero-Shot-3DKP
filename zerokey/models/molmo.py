"""Molmo multimodal model wrapper for keypoint detection.

Wraps the `Molmo <https://huggingface.co/allenai/Molmo-7B-D-0924>`_ family
of vision-language models to produce 2-D keypoint coordinates from a
natural-language prompt and an input image.  The model generates XML-style
``<point>`` elements whose ``x``/``y`` attributes are parsed into a dict
of named keypoints.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from matplotlib import colors as mcolors
from PIL import Image, ImageDraw
import molmo  # noqa: F401  # pyright: ignore[reportUnusedImport] — registers MolmoConfig/MolmoForCausalLM/MolmoProcessor with Auto classes
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig  # type: ignore[import-untyped]
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD  # type: ignore[import-untyped]
from xml.etree import ElementTree


class Molmo:
    """Thin wrapper around a Molmo VLM for keypoint generation.

    Loads the model and processor once on construction, then exposes
    :meth:`generated_kps_points` to generate keypoint text and
    :meth:`parse_points_str` / :meth:`draw_points` to decode and visualise
    the results.
    """

    COLOR_NAMES: Tuple[str, ...] = (
        'tab:red', 'tab:orange', 'tab:purple', 'tab:blue', 'tab:green',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    )
    COLOR_VALUES = tuple(
        map(mcolors.TABLEAU_COLORS.get, COLOR_NAMES)
    )

    def __init__(self, model_path: str = 'allenai/Molmo-7B-D-0924') -> None:
        """Load the Molmo model and processor.

        Args:
            model_path: HuggingFace model identifier.  Supported values
                include ``'allenai/MolmoE-1B-0924'``,
                ``'allenai/Molmo-7B-D-0924'`` (default), and
                ``'allenai/Molmo-72B-0924'``.
        """
        from transformers import AutoConfig  # type: ignore[import-untyped]
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=False,
            config=config,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map='auto',
        )

    @torch.inference_mode()
    def image_text_token(
        self, image: Image.Image, text: str = 'point to the salient keypoints in this image'
    ) -> Dict[str, Any]:
        """Tokenize an image–text pair for the model.

        Args:
            image: Input PIL image.
            text: Prompt string.

        Returns:
            Dict of tensors ready for ``model.generate_from_batch``.
        """
        inputs: Dict[str, Any] = self.processor.process(images=[image], text=text)
        return inputs

    @torch.inference_mode()
    def generated_kps_points(self, image: Image.Image, text: str) -> str:
        """Generate keypoint XML text from an image and a prompt.

        Args:
            image: Input PIL image.
            text: Natural-language instruction, e.g.
                ``"point to the armrest in this image"``.

        Returns:
            Raw generated string (typically XML ``<point>`` elements).
        """
        inputs = self.image_text_token(image, text=text)
        # Move to model device and add batch dimension
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(torch.bfloat16)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(  # type: ignore[operator]
                inputs,
                GenerationConfig(
                    max_new_tokens=2000,
                    stop_strings="<|endoftext|>",
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=False,
                    output_hidden_states=False,
                    output_attentions=False,
                ),
                tokenizer=self.processor.tokenizer,
            )

        generated_tokens = output.sequences[0, inputs['input_ids'].size(1):]
        generated_text: str = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True,
        )
        return generated_text

    @staticmethod
    def parse_points_str(input_str: str) -> Tuple[Dict[str, Dict[str, float]], str | None]:
        """Parse Molmo's XML output into a dict of keypoint coordinates.

        Args:
            input_str: XML string, e.g.
                ``'<points x1="23.4" y1="56.7" x2="..." alt="label">'``.

        Returns:
            ``(kps, alt)`` where *kps* maps point-id strings to
            ``{'x': float, 'y': float}`` dicts, and *alt* is the
            ``alt`` attribute (or ``None``).
        """
        root = ElementTree.fromstring(input_str)
        alt: str | None = root.attrib.get('alt', None)
        kps: Dict[str, Dict[str, float]] = defaultdict(dict)
        for k, v in root.attrib.items():
            if k.startswith(("x", "y")):
                kps[k[1:]][k[0]] = float(v)
        return kps, alt

    @staticmethod
    def draw_points(
        image: Image.Image,
        kps: Dict[str, Dict[str, float]],
        kps_wh: Tuple[float, float] = (100, 100),
        radius: int = 5,
        width: int | None = 2,
        colors: Sequence[str | None] = COLOR_VALUES,  # type: ignore[assignment]
    ) -> Image.Image:
        """Draw keypoint circles on *image* (in-place) and return it.

        Args:
            image: PIL image to draw on (modified in-place).
            kps: Mapping of point-id → ``{'x': …, 'y': …}`` in the
                coordinate system whose size is *kps_wh*.
            kps_wh: ``(width, height)`` of the coordinate system used
                by the keypoint values.
            radius: Circle radius in pixels.
            width: Outline width.  ``None`` fills the circle and builds
                a Gaussian-weighted alpha channel.
            colors: Sequence of colour strings, one per keypoint.

        Returns:
            The same *image* object, with circles drawn.
        """
        w, h = image.width, image.height
        k_w, k_h = kps_wh
        draw = ImageDraw.Draw(image)
        L = torch.zeros((h, w))

        for kp, color in zip(kps.values(), colors):
            x1 = float(kp['x']) / k_w * w
            y1 = float(kp['y']) / k_h * h

            if radius:
                if width:
                    draw.ellipse(
                        (x1 - radius, y1 - radius, x1 + radius, y1 + radius),
                        outline=color, width=width,
                    )
                else:
                    draw.ellipse(
                        (x1 - radius, y1 - radius, x1 + radius, y1 + radius),
                        fill=color,
                    )
                    alpha = Image.new(mode='L', size=(w, h))
                    ImageDraw.Draw(alpha).ellipse(
                        (x1 - radius, y1 - radius, x1 + radius, y1 + radius),
                        fill=1,
                    )
                    alpha_t = torch.from_numpy(np.asarray(alpha))
                    pos_y, pos_x = alpha_t.nonzero(as_tuple=True)
                    sigma = radius / 3
                    pdf = torch.sqrt((pos_x - x1) ** 2 + (pos_y - y1) ** 2)
                    pdf = torch.exp(-0.5 * (pdf / sigma) ** 2) / (sigma * math.sqrt(2 * torch.pi))
                    L[pos_y, pos_x] += pdf / torch.max(pdf)
            else:
                draw.point((x1, y1), fill=color)

        if radius and not width:
            L_img = Image.fromarray(
                torch.clamp(L * 255, 0, 255).to(dtype=torch.uint8).numpy()
            )
            image.putalpha(L_img)
        return image
