"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""
from pathlib import Path

import torch
from dataset import DayNightDataset
import sys
from utils.utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from utils.extract_BDD_images_list import get_list_of_images


def train_fn(
        disc_night, disc_day, gen_day, gen_night, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    """
    Training loop to train all the gan models.
    :param disc_night:
    :param disc_day:
    :param gen_day:
    :param gen_night:
    :param loader:
    :param opt_disc:
    :param opt_gen:
    :param l1:
    :param mse:
    :param d_scaler:
    :param g_scaler:
    """
    night_reals = 0
    night_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (day, night) in enumerate(loop):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_night = gen_night(day)
            disc_night_real = disc_night(night)
            disc_night_fake = disc_night(fake_night.detach())
            night_reals += disc_night_real.mean().item()
            night_fakes += disc_night_fake.mean().item()
            disc_night_real_loss = mse(disc_night_real, torch.ones_like(disc_night_real))
            disc_night_fake_loss = mse(disc_night_fake, torch.zeros_like(disc_night_fake))
            disc_night_loss = disc_night_real_loss + disc_night_fake_loss

            fake_day = gen_day(night)
            disc_day_real = disc_day(day)
            disc_day_fake = disc_day(fake_day.detach())
            disc_day_real_loss = mse(disc_day_real, torch.ones_like(disc_day_real))
            disc_day_fake_loss = mse(disc_day_fake, torch.zeros_like(disc_day_fake))
            disc_day_loss = disc_day_real_loss + disc_day_fake_loss

            # put it togethor
            total_disc_loss = (disc_night_loss + disc_day_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(total_disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            disc_night_fake = disc_night(fake_night)
            disc_day_fake = disc_day(fake_day)
            loss_gen_night = mse(disc_night_fake, torch.ones_like(disc_night_fake))
            loss_gen_day = mse(disc_day_fake, torch.ones_like(disc_day_fake))

            # cycle loss
            cycle_day = gen_day(fake_night)
            cycle_night = gen_night(fake_day)
            cycle_day_loss = l1(day, cycle_day)
            cycle_night_loss = l1(night, cycle_night)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_day = gen_day(day)
            identity_night = gen_night(night)
            identity_day_loss = l1(day, identity_day)
            identity_night_loss = l1(night, identity_night)

            # add all togethor
            G_loss = (
                    loss_gen_day
                    + loss_gen_night
                    + cycle_day_loss * config.LAMBDA_CYCLE
                    + cycle_night_loss * config.LAMBDA_CYCLE
                    + identity_night_loss * config.LAMBDA_IDENTITY
                    + identity_day_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_night * 0.5 + 0.5, f"saved_images/night_{idx}.png")
            save_image(fake_day * 0.5 + 0.5, f"saved_images/day_{idx}.png")

        loop.set_postfix(night_real=night_reals / (idx + 1), night_fake=night_fakes / (idx + 1))


def main():
    """
    main function to run the script.
    """
    disc_night = Discriminator(in_channels=3).to(config.DEVICE)
    disc_day = Discriminator(in_channels=3).to(config.DEVICE)
    gen_day = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_night = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_night.parameters()) + list(disc_day.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_day.parameters()) + list(gen_night.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_NIGHT,
            gen_night,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_DAY,
            gen_day,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_NIGHT,
            disc_night,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_DAY,
            disc_day,
            opt_disc,
            config.LEARNING_RATE,
        )
    train_labels_path = Path("C:/Others/BDD/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json")
    val_labels_path = Path("C:/Others/BDD/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json")
    list_of_day_images_train = get_list_of_images(weather_type="clear", timeofday="daytime",
                                                  json_labels_file_path=train_labels_path)
    list_of_night_images_train = get_list_of_images(weather_type="clear", timeofday="night",
                                                    json_labels_file_path=train_labels_path)
    list_of_day_images_val = get_list_of_images(weather_type="clear", timeofday="day",
                                                json_labels_file_path=val_labels_path)
    list_of_night_images_val = get_list_of_images(weather_type="clear", timeofday="night",
                                                  json_labels_file_path=val_labels_path)

    train_dataset = DayNightDataset(
        images_dir=config.IMAGES_DIR,
        transform=config.transforms,
        list_of_day_images=list_of_day_images_train,
        list_of_night_images=list_of_night_images_train
    )
    val_dataset = DayNightDataset(
        images_dir=config.IMAGES_DIR,
        transform=config.transforms,
        list_of_day_images=list_of_day_images_val,
        list_of_night_images=list_of_night_images_val
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_night,
            disc_day,
            gen_day,
            gen_night,
            train_loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_night, opt_gen, filename=config.CHECKPOINT_GEN_NIGHT)
            save_checkpoint(gen_day, opt_gen, filename=config.CHECKPOINT_GEN_DAY)
            save_checkpoint(disc_night, opt_disc, filename=config.CHECKPOINT_DISC_NIGHT)
            save_checkpoint(disc_day, opt_disc, filename=config.CHECKPOINT_DISC_DAY)


if __name__ == "__main__":
    main()
