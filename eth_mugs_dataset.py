import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from utils import IMAGE_SIZE, load_mask  # Ensure this import is correct


class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        self.image_paths = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.jpg')])
        self.mask_paths = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('_mask.png')])

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
           # transforms.Normalize((0.5,), (0.5,))
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print("[INFO] Dataset mode:", mode)
        print("[INFO] Number of images in the ETHMugsDataset: {}".format(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.rgb_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image = Image.open(image_path).convert('RGB')
        mask = load_mask(mask_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = Image.fromarray(mask.astype(np.uint8))
            mask = self.mask_transform(mask)

        return image, mask
