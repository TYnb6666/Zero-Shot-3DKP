"""GPT-4o visualization demo: query GPT-4o for keypoint locations and draw them on an image."""

import json
import os
import sys
from typing import Any

from openai import OpenAI
from PIL import Image

from zerokey.models import Molmo
from zerokey.generators.gpt4o import pil_to_base64, norm_location_list


def get_kplist(image: Image.Image) -> Any:
    """Query GPT-4o to localize a keypoint in the image and return the raw API response."""
    client = OpenAI()
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
                    "text": "Point to corner of the back of the chair in this image. Output the point location in the image (xy) coordinate system"
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


def main(image_path: "str | os.PathLike[str] | None" = None) -> None:
    """Run the GPT-4o visualization demo: query keypoints and draw them on a sample image."""
    if image_path is None:
        if len(sys.argv) < 2:
            raise SystemExit("Usage: python -m zerokey.vis.gpt4o <image_path>")
        image_path = sys.argv[1]
    image = Image.open(image_path)
    w, h = image.width, image.height
    response = get_kplist(image)
    content = json.loads(response.choices[0].message.content)
    kps = {k: v for k, v in norm_location_list(content)}
    print(kps)
    Molmo.draw_points(image, kps, kps_wh=(w, h))

    image.show()  # Display the image
