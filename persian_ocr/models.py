"""Small image classifier and two explicit comparison candidates."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CompactCNN(nn.Module):
    def __init__(self, classes: int = 32):
        super().__init__()
        layers = []
        previous = 1
        for width in (32, 64, 128):
            layers.extend([nn.Conv2d(previous, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            previous = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def mobilenet_model(*, pretrained: bool = False) -> nn.Module:
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, 32))
    return model


def model_inputs(images: torch.Tensor, name: str) -> torch.Tensor:
    if name != "mobilenet":
        return images
    # ImageNet expects RGB and normalized intensities. Preserve white background.
    images = torch.nn.functional.interpolate(1 - images, size=(128, 128), mode="bilinear", align_corners=False)
    images = images.repeat(1, 3, 1, 1)
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (images - mean) / std


def hog_features(images: np.ndarray) -> np.ndarray:
    from skimage.feature import hog
    return np.asarray([
        hog(image.squeeze(), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True)
        for image in images
    ], dtype=np.float32)
