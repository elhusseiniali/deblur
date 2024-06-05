import torch
from tqdm import tqdm
from config import device


class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.device = device

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_loader, val_loader, epochs):
        train_losses, validation_losses = (
            [],
            []
        )

        for i in range(epochs):
            print(f"Starting Epoch {i + 1} of {epochs}.")
            train_loss = self.train_step(train_loader)
            validation_loss = self.evaluate(val_loader)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            print(
                f"Epoch: {i+1}, Train loss: {train_loss:.4f}, "
                f"Validation loss: {validation_loss:.4f}"
            )

    def train_step(self, train_loader):
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
