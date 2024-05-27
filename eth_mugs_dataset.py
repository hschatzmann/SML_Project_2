import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, load_mask  # Ensure this import is correct

class ETHMugsDataset(Dataset):
    """Torch dataset template shared as an example."""

    def __init__(self, root_dir, mode="train"):
        """This dataset class loads the ETH Mugs dataset.

        It will return the resized image according to the scale and mask tensors
        in the original resolution.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): Mode of the dataset. It can be "train", "val" or "test"
        """
        self.mode = mode
        self.root_dir = root_dir

        # Get image and mask paths
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")

        self.image_paths = sorted([os.path.join(self.rgb_dir, f) for f in os.listdir(self.rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
        if self.mode != 'test':
            self.mask_paths = sorted([os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith('.png') or f.endswith('.jpg')])

        # Set image transforms
        self.transform = transforms.ToTensor()  # You can add more transformations as needed

        print("[INFO] Dataset mode:", mode)
        print("[INFO] Number of images in the ETHMugDataset: {}".format(len(self.image_paths)))

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        image = self.transform(image)

        if self.mode != 'test':
            # Load mask
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')  # Assuming mask is in grayscale
            mask = self.transform(mask)
            return image, mask
        else:
            return image

# Usage example:
# dataset = ETHMugsDataset(root_dir='./datasets/train_images_378_252', mode='train')
# image, mask = dataset[0]
