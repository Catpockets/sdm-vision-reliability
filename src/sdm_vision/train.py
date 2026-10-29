"""Training, prediction and temperature scaling, as in notebook 08."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ImageNetResNet


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Pixels:
    """Normalization with statistics from one run's training images, and the crops the network sees."""

    def __init__(self, train_images, device, crop=32):
        self.crop, self.device = crop, device
        self.mean = torch.tensor(train_images.mean(axis=(0, 2, 3)) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(train_images.std(axis=(0, 2, 3)) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def normalize(self, batch):
        return (batch.float() / 255 - self.mean) / self.std

    def augment(self, batch):
        # Random crop: pick a random crop x crop window for each image, then flip a random half left to right.
        size = self.crop
        n, _, h, w = batch.shape
        top = torch.randint(0, h - size + 1, (n, 1, 1), device=batch.device)
        left = torch.randint(0, w - size + 1, (n, 1, 1), device=batch.device)
        rows = top + torch.arange(size, device=batch.device).view(1, size, 1)
        cols = left + torch.arange(size, device=batch.device).view(1, 1, size)
        batch = batch[torch.arange(n, device=batch.device).view(n, 1, 1), :, rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(n, 1, 1, 1, device=batch.device) < 0.5
        return torch.where(flip, batch.flip(3), batch)

    def center_crop(self, batch):
        offset = (batch.shape[-1] - self.crop) // 2
        return batch[..., offset : offset + self.crop, offset : offset + self.crop]


def train_resnet(images, labels, num_classes, device, epochs=40, seed=0, batch_size=128, max_lr=0.1, weight_decay=5e-4, stem="cifar"):
    """Notebook 08's recipe: SGD with Nesterov momentum, one-cycle schedule, no label smoothing, final-epoch model."""
    torch.manual_seed(seed)
    pixels = Pixels(images, device)
    train_images, train_labels = torch.from_numpy(images).to(device), torch.from_numpy(labels).long().to(device)
    model = ImageNetResNet(num_classes=num_classes, stem=stem).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    steps_per_epoch = (len(train_images) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_images), device=device)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            loss = criterion(model(pixels.normalize(pixels.augment(train_images[batch]))), train_labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.eval()
    return model, pixels


@torch.no_grad()
def predict(model, pixels, images, batch_size=500):
    """Logits (CPU tensor) and 512-dimensional embeddings (NumPy) of the central crop of every image."""
    logits, embeddings = [], []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start : start + batch_size]).to(pixels.device)
        features = model.embed(pixels.normalize(pixels.center_crop(batch)))
        embeddings.append(features.cpu())
        logits.append(model.classifier(features).cpu())
    return torch.cat(logits), torch.cat(embeddings).numpy()


def fit_temperature(logits, labels):
    # The temperature that minimizes the negative log-likelihood, on a log-spaced grid from 0.5 to 5 (notebook 08).
    targets = torch.from_numpy(np.asarray(labels)).long()
    temperatures = np.exp(np.linspace(np.log(0.5), np.log(5), 200))
    nll = [F.cross_entropy(logits / t, targets).item() for t in temperatures]
    return float(temperatures[int(np.argmin(nll))])
