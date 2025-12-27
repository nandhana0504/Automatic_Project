import torch
import torch.nn as nn
from torchvision import models

class LandmarkModel(nn.Module):
    def __init__(self, num_landmarks=19):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_landmarks, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
