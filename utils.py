import matplotlib.pyplot as plt
import torch


def plot_sample(sample, figsize=(10, 5)):
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.imshow(sample[0].permute(*torch.arange(sample[0].ndim - 1, -1, -1)))
    plt.title('blur')

    plt.subplot(1, 2, 2)
    plt.imshow(sample[1].permute(*torch.arange(sample[0].ndim - 1, -1, -1)))
    plt.title('sharp')

    plt.show()


def plot_batch(dataloader, batch_index=0, figsize=(10, 5)):
    data_iter = iter(dataloader)

    for _ in range(batch_index):
        next(data_iter)
    batch = next(data_iter)

    for blur, sharp in zip(batch[0], batch[1]):
        plot_sample((blur, sharp), figsize)
