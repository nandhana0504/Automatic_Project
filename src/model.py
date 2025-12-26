import torch
import torch.nn as nn
from torchvision import models

class LandmarkModel(nn.Module):
    def __init__(self, num_landmarks=19):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.head = nn.Conv2d(512, num_landmarks, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)       # -> [B, 512, 8, 8]
        x = nn.functional.interpolate(x, size=(64, 64), mode="bilinear")
        x = self.head(x)          # -> [B, 19, 64, 64]
        return x
