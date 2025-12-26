import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CephDataset(Dataset):
    def __init__(self, image_dir, ann_dir, img_size=256, hm_size=64, sigma=2):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".bmp")])

    def gaussian(self, h, w, cx, cy):
        y, x = np.ogrid[:h, :w]
        return np.exp(-((x-cx)**2 + (y-cy)**2) / (2*self.sigma**2))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = cv2.imread(os.path.join(self.image_dir, img_name), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)

        ann_path = os.path.join(self.ann_dir, img_name.replace(".bmp", ".txt"))

        points = []
        with open(ann_path) as f:
            for line in f:
                if "," in line:
                    x, y = map(float, line.strip().split(","))
                    points.append([x, y])

        heatmaps = np.zeros((len(points), self.hm_size, self.hm_size), dtype=np.float32)

        for i, (x, y) in enumerate(points):
            hx = int(x * self.hm_size / w)
            hy = int(y * self.hm_size / h)
            heatmaps[i] = self.gaussian(self.hm_size, self.hm_size, hx, hy)

        return img, torch.tensor(heatmaps)
