"""ETH Mugs Dataset."""

import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, load_mask


# This is only an example - you DO NOT have to use it exactly in this form!
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

        # TODO: get image and mask paths
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        self.image_paths = os.path.join(self.root_dir, "rgb")  #TODO: what the fuck is this for? only len check? --> just use rgb again I guess

        # TODO: set image transforms - these transforms will be applied to pre-process the data before passing it through the model
        self.transform = transforms.ToTensor()  # TODO just to tensor rn, possibly normalize for better results later

        print("[INFO] Dataset mode:", mode)
        print(
            "[INFO] Number of images in the ETHMugDataset: {}".format(len(self.image_paths))
        )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        # TODO: load image and gt mask (unless when in test mode), apply transforms if necessary
        # TODO this means getting it, transforming it to tensor i guess?

        image = Image.open(self.rgb_dir + idx + "_rgb")     #open image, no conversion because should be in right format
        image = self.transform(image)                       #transform image

        mask = Image.open(self.mask_dir + idx +"_mask")     #open mask, no conversion because should be in right format
        mask = self.transform(mask)                         #transfrom mask

        return image, mask
