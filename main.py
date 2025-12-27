import os
import cv2
import torch
import numpy as np
from src.model import LandmarkModel

IMG_SIZE = 256
PIXEL_TO_MM = 0.1
THRESH_MM = 2.0

device = torch.device("cpu")
model = LandmarkModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

def get_coords(heatmaps):
    coords = []
    for hm in heatmaps:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords.append((x, y))
    return coords

correct = 0
total = 0

for name in sorted(os.listdir("data/images")):
    img = cv2.imread(f"data/images/{name}", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_t = torch.tensor(img_r).unsqueeze(0)
    img_t = img_t.repeat(3, 1, 1)
    img_t = img_t.unsqueeze(0).float()


    with torch.no_grad():
        pred = model(img_t)[0].numpy()

    pred_pts = get_coords(pred)

    gt_pts = []
    with open(f"data/annotations/{name.replace('.bmp','.txt')}") as f:
        for line in f.readlines()[:19]:
            x, y = map(float, line.split(','))
            x = x * IMG_SIZE / w
            y = y * IMG_SIZE / h
            gt_pts.append((x, y))

    for p, g in zip(pred_pts, gt_pts):
        dist_px = np.linalg.norm(np.array(p) - np.array(g))
        dist_mm = dist_px * PIXEL_TO_MM
        if dist_mm <= THRESH_MM:
            correct += 1
        total += 1

accuracy = correct / total * 100
print(f"🎯 Accuracy within 2mm: {accuracy:.2f}%")

if accuracy >= 90:
    print("✅ COMPANY REQUIREMENT MET")
else:
    print("⚠️ COMPANY REQUIREMENT NOT MET")
