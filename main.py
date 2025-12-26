import os
import cv2
import torch
import numpy as np
from src.model import LandmarkModel

IMAGE_DIR = "data/images"
ANN_DIR = "data/annotations"
MODEL_PATH = "outputs/model.pth"
IMAGE_SIZE = 256
MM_THRESHOLD = 2.0   # company requirement

def load_annotations(path):
    points = []
    with open(path) as f:
        for line in f:
            if "," in line:
                x, y = map(float, line.strip().split(","))
                points.append([x, y])
    return np.array(points)

def heatmap_to_coord(hm):
    idx = torch.argmax(hm)
    y = idx // hm.shape[1]
    x = idx % hm.shape[1]
    return x.item(), y.item()

def main():
    model = LandmarkModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    total = 0
    correct = 0

    for img_name in sorted(os.listdir(IMAGE_DIR)):
        print(f"🔍 Processing: {img_name}")

        img_path = os.path.join(IMAGE_DIR, img_name)
        ann_path = os.path.join(
            ANN_DIR, img_name.replace(".bmp", ".txt")
        )

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        gt = load_annotations(ann_path)

        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.tensor(
            img_resized / 255.0,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            heatmaps = model(img_tensor)[0]

        for i in range(len(gt)):
            px, py = heatmap_to_coord(heatmaps[i])

            px = px * w / IMAGE_SIZE
            py = py * h / IMAGE_SIZE

            gx, gy = gt[i]

            dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

            if dist <= MM_THRESHOLD:
                correct += 1

            total += 1

    accuracy = (correct / total) * 100
    print("✅ Prediction & evaluation completed")
    print(f"🎯 Accuracy within 2 mm: {accuracy:.2f}%")

    if accuracy >= 90:
        print("🎉 Acceptance criteria MET (≥ 90%)")
    else:
        print("⚠️ Acceptance criteria NOT met (< 90%)")

if __name__ == "__main__":
    main()
