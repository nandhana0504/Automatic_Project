import torch
from torch.utils.data import DataLoader
from src.dataset import CephDataset
from src.model import LandmarkModel
import os

def train():
    dataset = CephDataset(
        "data/images",
        "data/annotations",
        sigma=2
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = LandmarkModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    model.train()

    for epoch in range(150):
        total_loss = 0
        for img, hm in loader:
            pred = model(img)
            loss = loss_fn(pred, hm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/150] Loss: {total_loss:.4f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/model.pth")
    print("✅ Model saved")

if __name__ == "__main__":
    train()
