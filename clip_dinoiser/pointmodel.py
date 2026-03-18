import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import ImageDraw
from einops import rearrange
from hydra import compose, initialize_config_dir
from matplotlib import colors as mcolors
from torchvision import transforms as T

from .models.builder import build_model
from .segmentation.datasets.pascal_context import PascalContextDataset

_PKG_DIR = Path(__file__).resolve().parent

initialize_config_dir(config_dir=str(_PKG_DIR / "configs"), version_base=None)


def list_of_strings(arg):
    return arg.split(',')


class ClipDINOiser:
    PALETTE = list(PascalContextDataset.PALETTE)

    def __init__(self):
        self.cfg = compose(config_name='clip_dinoiser.yaml')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = _PKG_DIR / 'checkpoints' / 'last.pt'
        assert os.path.isfile(self.checkpoint_path), "Checkpoint file doesn't exist"
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

    def load_model_against_prompts(self, prompts):
        if len(prompts) == 1:
            prompts = ['background'] + prompts
        model = build_model(self.cfg.model, class_names=prompts)
        model.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
        model.eval()
        model.to(self.device)
        if 'background' in prompts:
            model.apply_found = True
        else:
            model.apply_found = False
        return model

    def visualize_per_image(self, images, masks, model: torch.nn.Module):
        """
        Visualizes output segmentation mask and saves it alongside with the labels in a file in a given output directory.

        :param file_path: [str] path to the image file
        :param TEXT_PROMPTS: [list(str)] list of text prompts to use for segmentation
        :param model: [torch.nn.module] loaded model for inference
        :param device: either "cpu" or "cuda"
        :param output_dir: [str] output directory
        :return:
        """

        img_tens = torch.cat([T.PILToTensor()(img).unsqueeze(0).to(self.device) / 255. for img in images], dim=0)

        h, w = img_tens.shape[-2:]

        output = model(img_tens)
        output = F.interpolate(output, scale_factor=model.vit_patch_size, mode="bilinear",  # type: ignore[arg-type]
                               align_corners=False)[..., :h, :w]
        sel = torch.stack(
            [torch.full_like(masks, fill_value=v, device=output.device, dtype=torch.int) for v in
             range(output.size(1))], dim=1)
        sel_mask = torch.logical_and(sel == torch.argmax(output, dim=1).unsqueeze(1),
                                     masks.unsqueeze(1).to(device=output.device))
        output = rearrange(output * sel_mask, 'B C H W -> B C (H W)')
        mask = torch.zeros_like(output, dtype=torch.bool)
        src = torch.ones((1, 1, 1), dtype=mask.dtype, device=mask.device).expand(*mask.shape[:-1], 1)
        mask.scatter_(dim=-1, index=torch.argmax(output, dim=-1, keepdim=True), src=src)
        mask = rearrange(torch.any(rearrange(sel_mask, 'B C H W -> B C (H W)'), dim=-1, keepdim=True) * mask,
                         'B C (H W) -> B C H W', H=w, W=w)
        return mask.cpu()

    def draw_points(self, image, kps, radius=5, width=2, colors=mcolors.TABLEAU_COLORS.values()):
        w, h = image.width, image.height
        for kp, color in zip(kps.nonzero(), colors):
            y1 = kp[0] / kps.shape[0] * h
            x1 = kp[1] / kps.shape[1] * w

            # Create an ImageDraw object
            draw = ImageDraw.Draw(image)
            color_str = str(color)

            if radius:
                # Draw circles at the specified points (eyes)
                if width:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), outline=color_str, width=width)
                else:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=color_str)
            else:
                draw.point((x1, y1), fill=color_str)
