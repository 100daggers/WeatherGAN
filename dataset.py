from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class DayNightDataset(Dataset):
    """
    Class DayNightDataset
    """

    def __init__(self, images_dir: Path, list_of_day_images: list[str], list_of_night_images: list[str], transform=None):
        self.images_dir = images_dir
        self.list_of_day_images = list_of_day_images
        self.list_of_night_images = list_of_night_images
        self.transform = transform

        self.length_dataset = max(len(self.list_of_day_images), len(self.list_of_night_images))  # 1000, 1500
        self.day_images_len = len(self.list_of_day_images)
        self.night_images_len = len(self.list_of_night_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        day_img = self.list_of_day_images[index % self.day_images_len]
        night_img = self.list_of_night_images[index % self.night_images_len]

        day_image_path = os.path.join(self.images_dir, day_img)
        night_image_path = os.path.join(self.images_dir, night_img)

        day_img = np.array(Image.open(day_image_path).convert("RGB"))
        night_img = np.array(Image.open(night_image_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=day_img, image0=night_img)
            day_img = augmentations["image"]
            night_img = augmentations["image0"]

        return day_img, night_img
