import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CephDataset(Dataset):
    def __init__(self, image_dir, anno_dir, img_size=256, sigma=5):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.img_size = img_size
        self.sigma = sigma
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def _gaussian_heatmap(self, center, size):
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = center
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        anno_path = os.path.join(self.anno_dir, img_name.replace('.bmp', '.txt'))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0).repeat(3, 1, 1)  # 1 → 3 channels


        coords = []
        with open(anno_path, 'r') as f:
            for line in f.readlines()[:19]:
                x, y = map(float, line.strip().split(','))
                x = x * self.img_size / w
                y = y * self.img_size / h
                coords.append((x, y))

        heatmaps = np.zeros((19, self.img_size, self.img_size), dtype=np.float32)
        for i, (x, y) in enumerate(coords):
            heatmaps[i] = self._gaussian_heatmap((x, y), self.img_size)

        return image, torch.tensor(heatmaps)
