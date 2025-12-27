import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import CephDataset
from src.model import LandmarkModel

def train():
    device = torch.device("cpu")

    dataset = CephDataset("data/images", "data/annotations")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = LandmarkModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 120
    for epoch in range(epochs):
        total_loss = 0
        for img, hm in loader:
            img, hm = img.to(device), hm.to(device)
            pred = model(img)
            loss = loss_fn(pred, hm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved")

if __name__ == "__main__":
    train()
