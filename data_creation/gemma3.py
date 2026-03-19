import io
import math
import os
import re
import time

import numpy as np
import PIL
from PIL import Image, ImageDraw

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import requests
import torch

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from einops import reduce

import sys


def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def read_image(url, target_size):
    contents = io.BytesIO(requests.get(url).content)
    image = Image.open(contents)
    image = crop_and_resize(image, target_size)
    image = np.array(image)
    # Remove alpha channel if necessary.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def parse_bbox_and_labels(detokenized_output: str):
  matches = re.finditer(
      '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
      ' (?P<label>.+?)( ;|$)',
      detokenized_output,
  )
  labels, boxes = [], []
  fmt = lambda x: float(x) / 1024.0
  for m in matches:
    d = m.groupdict()
    boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
    labels.append(d['label'])
  return np.array(boxes), np.array(labels)

def display_boxes(image, boxes, labels, target_image_size):
  h, l = target_image_size
  fig, ax = plt.subplots()
  ax.imshow(image)
  for i in range(boxes.shape[0]):
      y, x, y2, x2 = (boxes[i]*h)
      width = x2 - x
      height = y2 - y
      # Create a Rectangle patch
      rect = patches.Rectangle((x, y),
                               width,
                               height,
                               linewidth=1,
                               edgecolor='r',
                               facecolor='r')
      # Add label
      plt.text(x, y, labels[i], color='red', fontsize=12)
      # Add the patch to the Axes
      ax.add_patch(rect)

  plt.show()
  
def draw_boxes(image, boxes, target_image_size):
    h, l = target_image_size
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    
    for i in range(boxes.shape[0]):
        y, x, y2, x2 = (boxes[i]*h)
        # Draw rectangle
        draw.rectangle([(x, y), (x2, y2)], fill='red')
    
    return image

def display_segment_output(image, bounding_box, segment_mask, target_image_size):
    # Initialize a full mask with the target size
    target_width, target_height = target_image_size
    full_mask = np.zeros((target_height, target_width), dtype=np.uint8)

    for bbox, mask in zip(bounding_box, segment_mask):
        y1, x1, y2, x2 = bbox
        x1 = int(x1 * target_width)
        y1 = int(y1 * target_height)
        x2 = int(x2 * target_width)
        y2 = int(y2 * target_height)

        # Ensure mask is 2D before converting to Image
        if mask.ndim == 3:
            mask = mask.squeeze(axis=-1)
        mask = Image.fromarray(mask)
        mask = mask.resize((x2 - x1, y2 - y1), resample=Image.Resampling.NEAREST)
        mask = np.array(mask)
        binary_mask = (mask > 0.5).astype(np.uint8)


        # Place the binary mask onto the full mask
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], binary_mask)
    cmap = plt.get_cmap('jet')
    colored_mask = cmap(full_mask / 1.0)
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    if isinstance(image, Image.Image):
        image = np.array(image)
    blended_image = image.copy()
    mask_indices = full_mask > 0
    alpha = 0.5

    for c in range(3):
        blended_image[:, :, c] = np.where(mask_indices,
                                          (1 - alpha) * image[:, :, c] + alpha * colored_mask[:, :, c],
                                          image[:, :, c])

    fig, ax = plt.subplots()
    ax.imshow(blended_image)
    plt.show()
    return blended_image

def draw_segment_output(image, bounding_box, segment_mask, target_image_size):
    # Initialize a full mask with the target size
    target_width, target_height = target_image_size
    full_mask = np.zeros((target_height, target_width), dtype=np.uint8)

    for bbox, mask in zip(bounding_box, segment_mask):
        y1, x1, y2, x2 = bbox
        x1 = int(x1 * target_width)
        y1 = int(y1 * target_height)
        x2 = int(x2 * target_width)
        y2 = int(y2 * target_height)

        # Ensure mask is 2D before converting to Image
        if mask.ndim == 3:
            mask = mask.squeeze(axis=-1)
        mask = Image.fromarray(mask)
        mask = mask.resize((x2 - x1, y2 - y1), resample=Image.Resampling.NEAREST)
        mask = np.array(mask)
        binary_mask = (mask > 0.5).astype(np.uint8)

        # Place the binary mask onto the full mask
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], binary_mask)

    # Create colored mask using numpy operations instead of matplotlib
    colored_mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    colored_mask[full_mask > 0] = [255, 0, 0]  # Red color for mask

    if isinstance(image, Image.Image):
        image = np.array(image)
    blended_image = image.copy()
    mask_indices = full_mask > 0
    alpha = 0.5

    for c in range(3):
        blended_image[:, :, c] = np.where(mask_indices,
                                         (1 - alpha) * image[:, :, c] + alpha * colored_mask[:, :, c],
                                         image[:, :, c])

    return blended_image

class Gemma3:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        # load the processor
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map='auto')
        self.processor = AutoProcessor.from_pretrained(model_id, device_map='auto')


    @torch.inference_mode()
    def image_text_token(self, image, text='point to the salient keypoints in this image'):
        # process the image and text
        inputs = self.processor(text=text, images=image, return_tensors="pt").to("cuda")

        return inputs

    @torch.inference_mode()
    def generated_kps_points(self, image, text):
        inputs = self.image_text_token(image, text=text)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate(**inputs, max_new_tokens=200, use_cache=False)

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        return generated_text
        # >>> This image features an adorable black Labrador puppy sitting on a wooden deck.
        #     The puppy is positioned in the center of the frame, looking up at the camera...

    @staticmethod
    def draw_points(image, kps, radius=None, width=None, colors=None):
        pass


def main():
    gemma = Gemma3()
    image = Image.open(sys.argv[1]) if len(sys.argv) > 1 else Image.open(
        requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)  # type: ignore[arg-type]
    t0 = time.time()
    generated_text = gemma.generated_kps_points(image, text='<start_of_image> point to the armrest')
    t1 = time.time()
    print(t1 - t0)
    print(generated_text)


if __name__ == "__main__":
    main()
