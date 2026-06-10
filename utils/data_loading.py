import cv2
import numpy as np
import os

from pathlib import Path
from torch.utils.data import Dataset

class PetDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_names = sorted([
            f for f in os.listdir(image_dir) if f.endswith("jpg")
        ])
        self.dim = image_size

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir[index], self.file_names[index])
        mask_path = os.path.join(
            self.mask_dir, 
            self.file_names[index].replace("jpg", "png")
        )

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        return image, mask
    
    def _load_image(self, image_path: Path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize((image, self.dim, self.dim))

        return image

    def _load_mask(self, mask_path: Path):
        mask = np.array(cv2.imread(mask_path))
        mask = mask.astype(np.float32)
        mask = mask.resize(mask, (self.dim, self.dim))

        binary_mask = np.where(mask == 2, 0.0, 1.0)
        return binary_mask