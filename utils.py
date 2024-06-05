import matplotlib.pyplot as plt
import torch

import numpy as np


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def plot_sample(sample, figsize=(10, 5), normalized=False):
    plt.figure(figsize=figsize)

    num_cols = len(sample)
    plt.subplot(1, num_cols, 1)
    blurry_image = sample[0].permute(*torch.arange(sample[0].ndim - 1, -1, -1))
    if normalized:
        blurry_image = (blurry_image * std) + mean
    plt.imshow(blurry_image)
    plt.title('blurry image')

    plt.subplot(1, num_cols, 2)
    sharp_image = sample[1].permute(*torch.arange(sample[1].ndim - 1, -1, -1))
    if normalized:
        sharp_image = (sharp_image * std) + mean
    plt.imshow(sharp_image)
    plt.title('sharp image')

    if num_cols == 3:
        plt.subplot(1, num_cols, 3)
        prediction = sample[2].permute(*torch.arange(sample[2].ndim - 1, -1, -1))
        if normalized:
            prediction = (prediction * std) + mean
        plt.imshow(prediction)
        plt.title('model prediction')

    plt.show()


def plot_batch(dataloader, batch_index=0, figsize=(10, 5)):
    data_iter = iter(dataloader)

    for _ in range(batch_index):
        next(data_iter)
    batch = next(data_iter)

    for blur, sharp in zip(batch[0], batch[1]):
        plot_sample((blur, sharp), figsize)
