"""
This file is used to define all the hyperparameters required for training
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_DIR = "C:/Others/BDD/bdd100k_images_100k/bdd100k/images/100k/train_val"
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 2
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 60
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_NIGHT = "gen_night_with_identity"
CHECKPOINT_GEN_DAY = "gen_day_with_identity"
CHECKPOINT_DISC_NIGHT = "disc_night_with_identity"
CHECKPOINT_DISC_DAY = "disc_day_with_identity"

train_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

test_transforms = A.Compose(
    [
        A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
)
