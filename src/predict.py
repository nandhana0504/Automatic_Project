import os
import cv2
import torch
import numpy as np
from src.model import LandmarkModel

NUM_LANDMARKS = 19
IMG_SIZE = 512
HEATMAP_SIZE = 128

def heatmap_to_coord(hm):
    idx = hm.reshape(-1).argmax()
    y, x = divmod(idx, HEATMAP_SIZE)
    return x, y

def predict(image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkModel().to(device)
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
    model.eval()

    preds = {}

    for name in sorted(os.listdir(image_dir)):
        if not name.endswith(".bmp"):
            continue

        img = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        tensor = torch.tensor(img_r/255.0).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            hms = model(tensor)[0].cpu().numpy()

        coords = []
        for i in range(NUM_LANDMARKS):
            x, y = heatmap_to_coord(hms[i])
            coords.append([x/HEATMAP_SIZE*w, y/HEATMAP_SIZE*h])

        preds[name] = np.array(coords)

    return preds
