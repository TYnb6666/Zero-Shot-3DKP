"""Demo pipeline: use GPT-4o for keypoint naming and Molmo for 2D localization on a single image."""

import os
import sys
from xml.etree.ElementTree import ParseError

import json
from PIL import Image
from tqdm import tqdm

from zerokey.models import GPT4o, Molmo


def main(image_path: "str | os.PathLike[str] | None" = None) -> None:
    """Run the end-to-end demo: GPT-4o names keypoints, Molmo localizes and draws them."""
    if image_path is None:
        if len(sys.argv) < 2:
            raise SystemExit("Usage: python -m zerokey.vis.demo <image_path>")
        image_path = sys.argv[1]
    image = Image.open(image_path)
    gpt = GPT4o()
    response = gpt.get_kplist(image)
    message_content = response.choices[0].message.content
    assert message_content is not None
    content = json.loads(message_content)
    kp_list = [kp for kp in gpt.iter_over_list(content)]

    molmo = Molmo()
    print(','.join(kp_list), flush=True)
    kp_list = tqdm(kp_list)
    for kp in kp_list:
        kp_list.set_description(kp)
        kps = molmo.generated_kps_points(image, text=f"point to the {kp} in this image")
        try:
            kp_list.clear()
            print(kps)
            kps, alt = molmo.parse_points_str(kps)
            if len(kps) >= 5:
                print(f'too many kps for {alt}', file=sys.stderr)
                continue
            print(f"Drawing {alt} in this image", flush=True)
            molmo.draw_points(image, kps)
        except ParseError as e:
            print(e, file=sys.stderr)

    image.show()  # Display the image
