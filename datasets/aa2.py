from __future__ import annotations

import pickle
import random
import torch

from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from typing import Callable, Literal
from torchvision import datasets
import numpy as np


class AA2(Dataset):
    """
    AA2 Dataset
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
        download: bool = False,
        *args,
        **kwargs,
    ):
        self.root = root

        self.dataset_dir = (Path(root) / self.__class__.__name__).resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        resource_paths = (
            self.dataset_dir / "Animals_with_Attributes2",
            self.dataset_dir / "AwA2-base.zip",
            self.dataset_dir / "AwA2-data.zip",
        )
        if not all(path.exists() for path in resource_paths):
            # Download data
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it."
                )
        if split == "train":
            ds_idx = np.load(
                self.dataset_dir / "Animals_with_Attributes2" / "train_idx.npy"
            ).astype(int)
        elif split == "val":
            ds_idx = np.load(
                self.dataset_dir / "Animals_with_Attributes2" / "val_idx.npy"
            ).astype(int)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        import os

        if os.path.exists("/dev/shm/AA2/Animals_with_Attributes2/"):
            if os.path.exists(
                "/dev/shm/AA2/Animals_with_Attributes2/JPEGImages_optimized/"
            ):
                subfolder = "JPEGImages_optimized"
            else:
                subfolder = "JPEGImages"
            ds_all = datasets.ImageFolder(
                root=Path("/dev/shm/AA2/Animals_with_Attributes2/") / subfolder,
                transform=transform,
                *args,
                **kwargs,
            )
        else:

            ds_all = datasets.ImageFolder(
                root=self.dataset_dir / "Animals_with_Attributes2" / "JPEGImages",
                transform=transform,
                *args,
                **kwargs,
            )
        self.dataset = torch.utils.data.Subset(ds_all, ds_idx)
        self.split = split

        def load_animals(filename):
            animals = {}
            with open(filename, "r") as file:
                for line in file:
                    # Split each line at the tab characterd
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        # Replace '+' with spaces in the animal name
                        animal_name = parts[1].replace("+", " ")
                        animals[int(parts[0])] = animal_name
            return animals

        # Load the animals dictionary from the file
        animals_dict = load_animals(
            self.dataset_dir / "Animals_with_Attributes2" / "classes.txt"
        )
        self.animals_class_to_idx = {v: int(k) - 1 for k, v in animals_dict.items()}
        predicates_dict = load_animals(
            self.dataset_dir / "Animals_with_Attributes2" / "predicates.txt"
        )
        self.predicates_name_to_idx = {
            v: int(k) - 1 for k, v in predicates_dict.items()
        }
        modified_idx_to_class = {
            v: k.replace("+", " ") for k, v in ds_all.class_to_idx.items()
        }
        self.idx_from_pytorch_to_true = {
            k: self.animals_class_to_idx[v] for k, v in modified_idx_to_class.items()
        }

        self.predicate_matrix = np.loadtxt(
            self.dataset_dir
            / "Animals_with_Attributes2"
            / "predicate-matrix-binary.txt"
        )
        selected_predicate_names = [
            "forager",
            "white",
            "solitary",
            "small",
            "fierce",
            "plains",
        ]

        # Create an array of indices for these predicates
        selected_indices = [
            self.predicates_name_to_idx[name] for name in selected_predicate_names
        ]
        self.selected_predicates = selected_predicate_names
        self.selected_predicates_indices = selected_indices
        self.predicate_matrix = self.predicate_matrix[:, selected_indices]

    def download(self):
        """
        Download AA2
        """
        URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
        download_and_extract_archive(URL, self.dataset_dir, filename="AwA2-data.zip")

        URL = "https://cvml.ista.ac.at/AwA2/AwA2-base.zip"
        download_and_extract_archive(URL, self.dataset_dir, filename="AwA2-base.zip")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, pytorch_label = self.dataset[idx]
        label = self.idx_from_pytorch_to_true[pytorch_label]

        return (image, self.get_concepts(label)), label

    def get_concepts(self, label: int):

        # Define which predicates you want to keep
        return torch.tensor(self.predicate_matrix[label]).float()

    # def get_concepts(self, label: int):
    #     superclass_idx = LABEL_TO_SUPERCLASS[CLASSES[label]]
    #     return one_hot(torch.tensor(superclass_idx), len(SUPERCLASSES)).float()
