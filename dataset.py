import cv2
import numpy as np
import os
import random

from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

class FLAIRDataset(Dataset):
    """
    This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
    The images were obtained from The Cancer Imaging Archive (TCIA).

    Dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data
    """
    def __init__(
        self, 
        images_dir,
        subset,
        image_size=256,
        validation_cases=10,
        random_sampling=True,
        seed=111,
    ):
        random.seed(seed)    

        # Load images and masks
        volumes = {}
        masks = {}
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: x.split(".")[0].split("_")[4],
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    mask = mask.astype(np.float32)
                    mask_slices.append(mask)
                else:
                    image = cv2.imread(filepath, cv2.IMREAD_COLOR_RGB)
                    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
                    image_slices.append(image)

            if (image_slices):
                patient = dirpath.split("/")[-1]
                volumes[patient] = np.array(image_slices[1:-1])
                masks[patient] = np.array(mask_slices[1:-1])
        
        self.patients = sorted(volumes.keys())

        # Select cases (100 train, 10 validation cases)
        validation_patients = random.sample(self.patients, k=validation_cases)
        if subset == "validation":
            self.patients = validation_patients
        else:
            self.patients = list(set(self.patients).difference(validation_patients))

        self.volumes = [(volumes[k], masks[k]) for k in self.patients]
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]
        
        # Create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, _ in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling
        self.seed = seed
        self.image_size = image_size

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, index):
        p_idx = self.patient_slice_index[index][0]
        s_idx = self.patient_slice_index[index][1]

        v, m = self.volumes[p_idx]
        # v = self._normalize_volume(v)
    
        image = v[s_idx]
        mask = m[s_idx]

        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),

        ]
        transform = transforms.Compose(transform_list)
        image = transform(image)
        mask = transform(mask)

        return image, mask

    def _normalize_volume(self, volumes):
        mean = np.mean(volumes, axis=(0, 1, 2))
        std = np.std(volumes, axis=(0, 1, 2))
        volumes = (volumes - mean) / std

        return volumes
    