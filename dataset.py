from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor


class BlurAndSharp(Dataset):
    def __init__(self, path, img_size, image_limit=None, augment=False):
        """BlurandSharp constructor.

        Parameters
        ----------
        path : [pathlib.Path]
            Path to images to be used to construct the dataset.
        img_size : [int]
            Size of the image, assumed to be one of W or H, since images
            will be cropped to squares.
            e.g. if your image is (299, 299), you just set this to be 299.
            It will be converted to a tuple inside the constructor.
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
        self.root_dir = Path(path)
        self.img_size = (img_size, img_size)

        if augment:
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
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
            ])
        if image_limit:
            if not isinstance(image_limit, int):
                raise TypeError('image_limit should be an int.')

        self.blur_directory = (
            list(self.root_dir.joinpath('blur').glob('*.png')) +
            list(self.root_dir.joinpath('blur').glob('*.jpg')) +
            list(self.root_dir.joinpath('blur').glob('*.jpeg'))
        )
        self.blur_directory = self.blur_directory[:image_limit]

        self.sharp_directory = (
            list(self.root_dir.joinpath('sharp').glob('*.png')) +
            list(self.root_dir.joinpath('sharp').glob('*.jpg')) +
            list(self.root_dir.joinpath('sharp').glob('*.jpeg'))
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


def get_loader(path, img_size, batch_size, image_limit=False, augment=True):
    dataset = BlurAndSharp(
        path=path,
        img_size=img_size,
        image_limit=image_limit,
        augment=True
    )
    loader = DataLoader(dataset, batch_size, shuffle=True)
    return loader
