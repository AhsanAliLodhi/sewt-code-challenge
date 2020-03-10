from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import os
from skimage import transform
from PIL import Image
from torchvision import transforms, utils


class TowelsDataset(Dataset):
    """Towels dataset."""

    def __init__(self, csv_file, root_dir, transform=None,
                 custom_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.custom_transform = custom_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        bbox = np.array(self.labels.iloc[idx, 1:]).astype('float')
        # transform it to x, y, w, h
        sample = {'image': image, 'bbox': np.array([bbox[0], bbox[1],
                  bbox[2], bbox[3]]), 'image_name': img_name}

        if self.custom_transform:
            sample = self.custom_transform(sample)
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
