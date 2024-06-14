from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor


class BlurAndSharp(Dataset):
    def __init__(self, data_path, config, image_limit=None, augment=False):
        """BlurandSharp constructor.

        Parameters
        ----------
        data_path : [pathlib.Path]
            Path to images to be used to construct the dataset.
        config: [Config]
            Config to use.
        image_limit : [int or None], optional
            How many images the dataset should containt, by default None
        augment : [bool], optional
            Whether or not to use data augmentation, by default False

        Raises
        ------
        TypeError
            If image_limit is not an int
        ValueError
            If the number of blurry and sharp images isn't the same
        """
        self.data_path = Path(data_path)
        self.config = config
        img_size = self.config.img_size
        self.img_size = (img_size, img_size)

        if augment:
            mean = self.config.mean
            std = self.config.std

            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.2),
                    transforms.Normalize(
                        mean=mean,
                        std=std
                    ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
            ])
        if image_limit:
            if not isinstance(image_limit, int):
                raise TypeError('image_limit should be an int.')

        self.blur_directory = (
            list(self.data_path.joinpath('blur').glob('*.png')) +
            list(self.data_path.joinpath('blur').glob('*.jpg')) +
            list(self.data_path.joinpath('blur').glob('*.jpeg'))
        )
        self.blur_directory = self.blur_directory[:image_limit]

        self.sharp_directory = (
            list(self.data_path.joinpath('sharp').glob('*.png')) +
            list(self.data_path.joinpath('sharp').glob('*.jpg')) +
            list(self.data_path.joinpath('sharp').glob('*.jpeg'))
        )
        self.sharp_directory = self.sharp_directory[:image_limit]
        if len(self.blur_directory) != len(self.sharp_directory):
            raise ValueError(
                "You should have the same number of blurry and sharp images. "
                f"Found {len(self.blur_directory)} blurry images, and "
                f"{len(self.sharp_directory)} sharp images.")

    def __len__(self):
        return len(self.blur_directory)

    def __getitem__(self, idx):
        blurry_image_path = self.blur_directory[idx]
        sharp_image_path = self.sharp_directory[idx]

        blurry_image = ToTensor()(
            Image.open(blurry_image_path).convert('RGB')
        )

        sharp_image = ToTensor()(
            Image.open(sharp_image_path).convert('RGB')
        )
        combined_images = self.transform(
            torch.cat(
                (
                    blurry_image.unsqueeze(0),
                    sharp_image.unsqueeze(0)
                ), 0)
        )

        return (combined_images[0], combined_images[1])


def get_loader(data_path, config, batch_size, image_limit=False, augment=True):
    dataset = BlurAndSharp(
        data_path=data_path,
        config=config,
        image_limit=image_limit,
        augment=augment
    )
    loader = DataLoader(dataset, batch_size, shuffle=True)
    return loader
