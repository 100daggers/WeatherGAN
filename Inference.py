"""
Inference for CycleGAN
"""
import os.path
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import save_image

import config
from config import test_transforms
from generator_model import Generator
from utils.utils import load_checkpoint


def save_sample(real: torch.Tensor, fake: torch.Tensor, path: Path, epoch: int, image_name: str) -> None:
    """
    concatenates the real and fake images horizontally and saves them in the input directory.
    :param image_name:
    :param epoch:
    :param path:
    :param real:
    :param fake:
    """
    print("-----saving_image-----")
    real_and_fake = torch.cat((real, fake), dim=2)
    save_image(real_and_fake, os.path.join(path, f"{image_name}_{epoch}.png"))


def main():
    """
    main function to run the script.
    
    """
    gen_day = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_night = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_day.parameters()) + list(gen_night.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    epoch_num = 59

    if config.LOAD_MODEL:
        load_checkpoint(
            f"gen_night_with_identity_epoch_{epoch_num}.pth.tar",
            gen_night,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            f"gen_day_with_identity_epoch_{epoch_num}.pth.tar",
            gen_day,
            opt_gen,
            config.LEARNING_RATE,
        )
    test_images_day_path = Path("C:/Others/BDD/bdd100k_images_100k/bdd100k/images/100k/test_sub_set_day")
    for idx, day_image in enumerate(test_images_day_path.glob("*.jpg")):
        name = day_image.name
        day_image = np.array(Image.open(day_image).convert("RGB"))
        augmentations = test_transforms(image=day_image)
        day_image = augmentations["image"].to(config.DEVICE)
        with torch.no_grad():
            fake_night = gen_night(day_image)
        save_sample(real=day_image.detach().cpu() * 0.5 + 0.5, fake=fake_night.detach().cpu() * 0.5 + 0.5,
                    path=Path("C:/Others/WeatherGAN_ADAS/inference/day_to_night"),
                    epoch=epoch_num, image_name=name)


if __name__ == "__main__":
    main()
