"""
Model wrappers for feature extraction from large pre-trained vision models.
"""
from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


class ModelWrapper(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, images: torch.Tensor, interpolation: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def patch_size(self) -> int:
        pass


class DINOWrapper(ModelWrapper):
    model: Any  # DINOv2 model with forward_features method

    def __init__(self, device: str = 'cpu', small: bool = False) -> None:
        super().__init__()
        if not small:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.to(device)
        self.model.eval()

        self.image_transforms = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, images: torch.Tensor, interpolation: bool = False) -> torch.Tensor:
        images = self.image_transforms(images.permute(0, 3, 1, 2))
        out = self.model.forward_features(images)
        if interpolation:
            features = out['x_norm_patchtokens']
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out['x_norm_patchtokens']

    def patch_size(self) -> int:
        return 14


class CLIPWrapper(ModelWrapper):
    def __init__(self, device: str = 'cpu') -> None:
        super().__init__()
        import clip
        self.model, _ = clip.load('ViT-L/14', device=device)
        self.model.eval()

        self.image_transforms = T.Compose([
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images: torch.Tensor, interpolation: bool = False) -> torch.Tensor:
        images = self.image_transforms(images.permute(0, 3, 1, 2)).type(self.model.dtype)
        x = self.model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],  # type: ignore[union-attr]
                                                                                   dtype=x.dtype, device=x.device), x],
                      dim=1)
        x = x + self.model.visual.positional_embedding.to(x.dtype)  # type: ignore[union-attr]
        x = self.model.visual.ln_pre(x)  # type: ignore[union-attr]

        x = x.permute(1, 0, 2)
        x = self.model.visual.transformer(x)  # type: ignore[union-attr]
        x = x.permute(1, 0, 2)

        out = self.model.visual.ln_post(x[:, :, :])  # type: ignore[union-attr]

        if self.model.visual.proj is not None:
            out = out @ self.model.visual.proj

        if interpolation:
            features = out[:, 1:, :]
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out[:, 1:, :]

    def patch_size(self) -> int:
        return 14


class SAMWrapper(ModelWrapper):
    def __init__(self, device: str = 'cpu') -> None:
        super().__init__()
        from segment_anything import sam_model_registry  # type: ignore[import-not-found]
        self.model = sam_model_registry["vit_l"](checkpoint="/users/wimmer/sam_vit_l_0b3195.pth").to(device)
        self.model.eval()

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - torch.tensor((123.675, 116.28, 103.53), device=self.model.device).view(-1, 1,
                                                                                                  1) / torch.tensor(
            (58.395, 57.12, 57.375), device=self.model.device).view(-1, 1, 1))
        h, w = images.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        return F.pad(images, (0, padw, 0, padh))

    def forward(self, images: torch.Tensor, interpolation: bool = False) -> torch.Tensor:
        images = self.preprocess(images)
        x = self.model.image_encoder.patch_embed(images)
        if self.model.image_encoder.pos_embed is not None:
            x = x + self.model.image_encoder.pos_embed
        for blk in self.model.image_encoder.blocks:
            x = blk(x)

        out = x.view(x.shape[0], -1, x.shape[-1])

        if interpolation:
            features = out
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out

    def patch_size(self) -> int:
        return 1


class EffNetWrapper(ModelWrapper):
    model: Any  # EfficientNet model with extract_features method

    def __init__(self, device: str = 'cpu') -> None:
        super().__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        self.model.eval()
        self.model.to(device)
        self.image_transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, images: torch.Tensor, interpolation: bool = False) -> torch.Tensor:
        images = self.image_transforms(images.permute(0, 3, 1, 2))
        out = self.model.extract_features(images)["layer5"]  # type: ignore[attr-defined]
        out = torch.permute(out, (0, 2, 3, 1)).view(out.shape[0], -1, out.shape[1])
        if interpolation:
            features = out
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out

    def patch_size(self) -> int:
        return 14  # TODO: CHECK THIS
