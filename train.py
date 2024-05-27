"""Code template for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from utils import dice_coeff, load_mask, compute_iou, IMAGE_SIZE
from eth_mugs_dataset import ETHMugsDataset


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = UNetBlock(in_channels, 64)
        self.encoder2 = UNetBlock(64, 128)
        self.encoder3 = UNetBlock(128, 256)
        self.encoder4 = UNetBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder4 = UNetBlock(1024, 512)
        self.decoder3 = UNetBlock(512, 256)
        self.decoder2 = UNetBlock(256, 128)
        self.decoder1 = UNetBlock(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self.decoder4(torch.cat([bottleneck, enc4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))

        # Output layer
        out = self.out(dec1)
        return out

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def build_model():  # TODO: Add your model definition here
    """Build the model."""
    #model = UNet(in_channels=3, out_channels=1)  # Adjust in_channels and out_channels as needed
    model = SimpleCNN(in_channels=3, out_channels=1)
    return model

def train(ckpt_dir: str, train_data_root: str, val_data_root: str):
    """Train function."""
    log_frequency = 10
    val_batch_size = 1
    val_frequency = 1

    num_epochs = 50
    lr = 1e-4
    train_batch_size = 8

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = ETHMugsDataset(root_dir=val_data_root, mode="val")
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    model = build_model()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_iou = 0

        for batch_idx, (image, gt_mask) in enumerate(train_dataloader):
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, gt_mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Dynamic thresholding
            threshold = output.mean()  # Use mean value as threshold
            pred_mask = (torch.sigmoid(output) > threshold).float()

            gt_mask_np = gt_mask.cpu().numpy().astype(int)
            pred_mask_np = pred_mask.cpu().numpy().astype(int)

            epoch_iou += compute_iou(pred_mask_np, gt_mask_np)

            if batch_idx % log_frequency == 0:
                print(
                    f"[INFO]: Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        lr_scheduler.step()
        print(f"[INFO]: Epoch [{epoch}/{num_epochs}], Average Loss: {epoch_loss / len(train_dataloader):.4f}")
        print(f"[INFO]: Epoch [{epoch}/{num_epochs}], Average IoU: {epoch_iou / len(train_dataloader):.4f}")


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

                    output = model(val_image)
                    pred_mask = torch.sigmoid(output) > 0.5

                    pred_mask_np = pred_mask.cpu().numpy().astype(int)
                    val_gt_mask_np = val_gt_mask.cpu().numpy().astype(int)

                    val_dice += dice_coeff(pred_mask, val_gt_mask).item()
                    val_iou += compute_iou(pred_mask_np, val_gt_mask_np)

                val_dice /= len(val_dataloader)
                val_iou /= len(val_dataloader)

                print(f"[INFO]: Validation Dice Coefficient: {val_dice:.4f}")
                print(f"[INFO]: Validation IoU: {val_iou:.4f}")

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))
        val_images, val_masks = next(iter(val_dataloader))
        model.eval()
        with torch.no_grad():
            val_images = val_images.to(device)
            val_outputs = model(val_images)
            val_outputs = torch.sigmoid(val_outputs)
            val_outputs = (val_outputs > 0.5).float()
        # Plot the image, ground truth mask, and predicted mask
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(val_images[0].cpu().permute(1, 2, 0))
        plt.title('Image')
        plt.subplot(1, 3, 2)
        plt.imshow(val_masks[0].cpu().squeeze(), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.subplot(1, 3, 3)
        plt.imshow(val_outputs[0].cpu().squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.show()


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
