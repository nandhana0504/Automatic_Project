import torch
import numpy as np
from src.dataset import CephDataset
from src.model import LandmarkModel

def evaluate():
    dataset = CephDataset("data/images", "data/annotations")
    model = LandmarkModel()
    model.load_state_dict(torch.load("landmark_model.pt"))
    model.eval()

    correct = 0
    total = 0

    for img, target in dataset:
        pred = model(img.unsqueeze(0)).detach().numpy()[0]
        target = target.numpy()

        pred = pred.reshape(-1, 2)
        target = target.reshape(-1, 2)

        dist = np.linalg.norm(pred - target, axis=1)

        correct += np.sum(dist < 0.01)  # ≈ 2mm normalized
        total += len(dist)

    acc = correct / total * 100
    print(f"🎯 Accuracy within 2mm: {acc:.2f}%")

if __name__ == "__main__":
    evaluate()
