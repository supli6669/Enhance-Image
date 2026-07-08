import os
import argparse
import glob
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class SimpleUNet(nn.Module):
    """Lightweight UNet‑style model for image sharpening (placeholder)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(dataset_dir, checkpoint_dir, epochs, batch_size, lr, use_gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SimpleUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Start training on {len(dataset)} images (device: {device})")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
    final_path = os.path.join(checkpoint_dir, "sharpening_model_final.pth")
    torch.save(model.state_dict(), final_path)
    print("Training completed. Model saved to", final_path)

def main():
    parser = argparse.ArgumentParser(description="Train sharpening model on 512x512 PNG images.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Folder containing PNG images.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to store checkpoints.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args.dataset_dir, args.checkpoint_dir, args.epochs, args.batch_size, args.lr, args.gpu)

if __name__ == "__main__":
    main()
