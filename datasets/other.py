from __future__ import annotations

import torch

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTModulo(Dataset):
    """
    MNIST Dataset with additional concepts for divisibility mod 5.

    Data: image

    Targets: digit from 0 to 9

    Concepts:
        * digit % 5 == 0
        * digit % 5 == 1
        * digit % 5 == 2
        * digit % 5 == 3
        * digit % 5 == 4
    """

    def __init__(self, root: str, train: bool = True):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        mnist = datasets.MNIST(
            root=root, train=train, transform=transform, download=True)

        self.data = mnist.data.float()
        self.targets = mnist.targets.long()
        self.concepts = torch.stack([
            self.targets % p == i for p in (5,) for i in range(p)], dim=-1).float()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data[idx], self.concepts[idx]), self.targets[idx]
