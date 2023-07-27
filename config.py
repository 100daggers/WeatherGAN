"""
This file is used to define all the hyperparameters required for training
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_DIR = "C:/Others/BDD/bdd100k_images_100k/bdd100k/images/100k/train_val"
IMAGE_SIZE = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_NIGHT = "gen_night.pth.tar"
CHECKPOINT_GEN_DAY = "gen_day.pth.tar"
CHECKPOINT_DISC_NIGHT = "disc_night.pth.tar"
CHECKPOINT_DISC_DAY = "disc_day.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
