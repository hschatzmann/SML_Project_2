"""Code template for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from utils import dice_coeff, load_mask, compute_iou, IMAGE_SIZE
rom eth_mugs_dataset import ETHMugsDataset


class UNet(nn.Module):  #Unet model
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expansive path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # Expansive path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)

def build_model():  # TODO: Add your model definition here
    """Build the model."""
    model = UNet(in_channels=3, out_channels=1)  # Adjust in_channels and out_channels as needed
    return model

def train(ckpt_dir: str, train_data_root: str, val_data_root: str):
    """Train function."""
    # Logging and validation settings
    log_frequency = 10
    val_batch_size = 1
    val_frequency = 1

    # Hyperparameters
    num_epochs = 50
    lr = 1e-4
    train_batch_size = 8

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")

    # Choose Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define your Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = ETHMugsDataset(root_dir=train_data_root, transform=transform, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = ETHMugsDataset(root_dir=val_data_root, transform=transform, mode="val")
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # Define your model
    model = build_model()
    model.to(device)

    # Define Loss function
    criterion = nn.BCEWithLogitsLoss()  # For binary segmentation

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define Learning rate scheduler if needed
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (image, gt_mask) in enumerate(train_dataloader):
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(image)
            output = torch.sigmoid(output)  # Apply sigmoid for BCEWithLogitsLoss
            loss = criterion(output, gt_mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % log_frequency == 0:
                print(f"[INFO]: Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        lr_scheduler.step()
        print(f"[INFO]: Epoch [{epoch}/{num_epochs}], Average Loss: {epoch_loss / len(train_dataloader):.4f}")

        if epoch % val_frequency == 0:
            model.eval()
            val_dice = 0.0
            val_iou = 0.0
            with torch.no_grad():
                for val_image, val_gt_mask in val_dataloader:
                    val_image = val_image.to(device)
                    val_gt_mask = val_gt_mask.to(device)

                    # Forward pass
                    output = model(val_image)
                    output = torch.sigmoid(output)  # Apply sigmoid for consistency

                    val_dice += dice_coeff(output, val_gt_mask).item()
                    val_iou += compute_iou(output.cpu().numpy(), val_gt_mask.cpu().numpy())

                val_dice /= len(val_dataloader)
                val_iou /= len(val_dataloader)

                print(f"[INFO]: Validation Dice Coefficient: {val_dice:.4f}")
                print(f"[INFO]: Validation IoU: {val_iou:.4f}")

        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="Path to save the model checkpoints to.",
    )
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    # Set data root
    train_data_root = os.path.join(args.data_root, "train_images_378_252")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "public_test_images_378_252")
    print(f"[INFO]: Validation data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)
