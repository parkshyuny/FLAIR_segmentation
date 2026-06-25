import cv2
import numpy as np
import os
import random
import torch

from torch.utils.data import Dataset
from torchvision import transforms

class FLAIRDataset(Dataset):
    """
    This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
    The images were obtained from The Cancer Imaging Archive (TCIA).

    Dataset is publicly available on Kaggle: 
    https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data
    """
    def __init__(
        self, 
        IMAGES_DIR,
        SUBSET,
        IMAGE_SIZE=256,
        VALIDATION_CASES=10,
        RANDOM_FLAG=True,
        SEED=111,
    ):
        self.random_sampling = RANDOM_FLAG
        self.seed = SEED
        self.image_size = IMAGE_SIZE
        random.seed(SEED)    
        
        
        volumes = {}
        masks = {}
        for (dirpath, _, filenames) in os.walk(IMAGES_DIR):
            image_slices = []
            mask_slices = []

            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames), key=lambda x: x.split(".")[0].split("_")[4],
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    mask_slices.append(mask)
                else:
                    image = cv2.imread(filepath, cv2.IMREAD_COLOR_RGB)
                    image_slices.append(image)

            if (image_slices):
                patient = dirpath.split("/")[-1]
                
                volumes[patient] = np.array(image_slices[1:-1])
                masks[patient] = np.array(mask_slices[1:-1])
        
        self.patients = sorted(volumes.keys())

        validation_patients = random.sample(self.patients, k=VALIDATION_CASES)
        if SUBSET == "valid":
            # 10 for validation
            self.patients = validation_patients
        else:
            # 100 for train
            self.patients = list(set(self.patients).difference(validation_patients))

        self.volumes = [(volumes[k], masks[k]) for k in self.patients]
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]
        
        # Create global index (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, _ in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, index):
        p_idx = self.patient_slice_index[index][0]
        s_idx = self.patient_slice_index[index][1]

        images, masks = self.volumes[p_idx]
    
        image = images[s_idx]
        mask = masks[s_idx]

        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
        ]
        transform = transforms.Compose(transform_list)
        
        image = transform(image)
        mask = transform(mask)

        return image, mask
    