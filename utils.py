import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_sample(sample, config, figsize=(10, 5), normalized=True):
    mean = torch.tensor(config.mean)
    std = torch.tensor(config.std)

    plt.figure(figsize=figsize)
    num_cols = len(sample)
    plt.subplot(1, num_cols, 1)
    blurry_image = sample[0].permute(
        *torch.arange(sample[0].ndim - 1, -1, -1)
    )
    if normalized:
        blurry_image = (blurry_image * std) + mean
    plt.imshow(blurry_image)
    plt.title('blurry image')

    plt.subplot(1, num_cols, 2)
    sharp_image = sample[1].permute(
        *torch.arange(sample[1].ndim - 1, -1, -1)
    )
    if normalized:
        sharp_image = (sharp_image * std) + mean
    plt.imshow(sharp_image)
    plt.title('sharp image')

    if num_cols == 3:
        plt.subplot(1, num_cols, 3)
        prediction = sample[2].permute(
            *torch.arange(sample[2].ndim - 1, -1, -1)
        )
        if normalized:
            prediction = (prediction * std) + mean
        plt.imshow(prediction)
        plt.title('model prediction')

    plt.show()


def plot_batch(dataloader, config, batch_index=0, figsize=(10, 5)):
    data_iter = iter(dataloader)

    for _ in range(batch_index):
        next(data_iter)
    batch = next(data_iter)

    for blur, sharp in zip(batch[0], batch[1]):
        plot_sample(sample=(blur, sharp), config=config, figsize=figsize)


def plot_losses(train_losses, validation_losses, model_id, experiment_name=''):
    epochs = len(train_losses)
    if len(validation_losses) != epochs:
        raise ValueError(
            'Number of losses for training and validation does not match.')
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Losses for {model_id}")
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    plt.xticks(range(1, epochs + 1))
    if experiment_name:
        figure_dir = Path('./figures')
        figure_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(
            figure_dir / f'{experiment_name}.png'
        )
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()


def save_model(model, model_id, experiment_name, epoch):
    checkpoint_dir = Path('./checkpoints')
    model_dir = Path(checkpoint_dir) / f'{model_id}'
    model_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        Path(model_dir) / f'{experiment_name}_epoch-{epoch}_model_weights.pth'
    )
    torch.save(model.state_dict(), output_path)
