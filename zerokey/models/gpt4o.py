"""GPT-4o multimodal model wrapper for keypoint list generation.

Queries the OpenAI ``gpt-4o`` chat endpoint with an image and a prompt,
requesting a JSON-formatted list of salient keypoint names.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Iterator

from openai import OpenAI  # type: ignore[import-untyped]
from PIL import Image


class GPT4o:
    """Thin wrapper around the OpenAI chat API for keypoint extraction."""

    @staticmethod
    def pil_to_base64(pil_image: Image.Image) -> str:
        """Encode a PIL image as a base64 PNG string."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def __init__(self) -> None:
        self.client = OpenAI()

    def get_kplist(self, image: Image.Image) -> Any:
        """Ask GPT-4o to list salient keypoints for *image*.

        Returns:
            The raw ``ChatCompletion`` response object.
        """
        image_encoded = self.pil_to_base64(image)
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
                        "text": "List possible salient keypoints (in text)"
                    }
                ]
            },
        ]
        response = self.client.chat.completions.create(
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

    def iter_over_list(self, content: Any) -> Iterator[Any]:
        """Recursively yield leaf values from a nested dict/list structure."""
        for _k, v in content.items():
            if isinstance(v, dict):
                yield from self.iter_over_list(v)
            elif isinstance(v, list | set | tuple):
                yield from v
            else:
                yield v
