import torch
from tqdm import tqdm

from config import SUPPORTED_MODELS
from utils import plot_sample, plot_losses


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        config,
        debug=False, debug_step=100, debug_image_count=1
    ):
        self.config = config
        self.device = self.config.device

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.debug = debug
        self.debug_step = debug_step
        self.debug_image_count = debug_image_count

    @staticmethod
    def supported_models():
        return list(SUPPORTED_MODELS)

    def train(self, train_loader, val_loader, epochs):
        train_losses, validation_losses = ([], [])

        for i in range(epochs):
            print(f"Starting Epoch {i + 1} of {epochs}.")
            flag = False
            if self.debug:
                if i == 0 or ((i + 1) % self.debug_step == 0):
                    flag = True
            train_loss = self.train_step(
                train_loader,
                debug=flag,
                debug_image_count=self.debug_image_count)
            validation_loss = self.evaluate(val_loader)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            print(
                f"Epoch: {i+1}, Train loss: {train_loss:.4f}, "
                f"Validation loss: {validation_loss:.4f}"
            )
        plot_losses(train_losses, validation_losses)

    def train_step(self, train_loader, debug=False, debug_image_count=1):
        self.model.train()
        train_loss = 0
        for blur, sharp in tqdm(train_loader, unit="batch", total=len(train_loader)):
            blurry_batch, sharp_batch = blur.to(self.device), sharp.to(self.device)
            img_shape = blurry_batch.shape[1:]
            # Zero the gradients
            self.optimizer.zero_grad()
            output_batch = self.model(blurry_batch).view(-1, *img_shape)
            # Compute loss for current batch
            loss = self.criterion(output_batch, sharp_batch)
            # Backpropagation
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            # Add loss to total loss
            train_loss += loss.item() * blur.size(0)
            if debug:
                if debug_image_count > len(blurry_batch):
                    debug_image_count = 1
                for idx in range(debug_image_count):
                    input_image = blurry_batch[idx].clone().detach().cpu()
                    label = sharp_batch[idx].clone().detach().cpu()
                    prediction = output_batch[idx].clone().detach().cpu()
                    plot_sample(
                        sample=(input_image, label, prediction),
                        config=self.config,
                        normalized=True,
                    )
        return train_loss

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0
        for blur, sharp in tqdm(val_loader, unit="batch", total=len(val_loader)):
            blurry_batch, sharp_batch = blur.to(self.device), sharp.to(self.device)
            img_shape = blurry_batch.shape[1:]

            output_batch = self.model(blurry_batch).view(-1, *img_shape)
            # Compute loss for current batch
            loss = self.criterion(output_batch, sharp_batch)

            # Add loss to total loss
            val_loss += loss.item() * blur.size(0)
        return val_loss
