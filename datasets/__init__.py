import functools
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from typing import Any

from .cifar import CIFAR100
from .cub import CUB
from .oai import OAI
from .other import MNISTModulo
from .pitfalls import MNIST_45, DatasetC, DatasetD, DatasetE
from .imagenet import ImageNet
from .celeba import generate_data as celeba_generate_data
from .aa2 import AA2
from .cxr import MIMIC_CXR, CheX_Dataset
import torchxrayvision as xrv
from .cifar import SUPERCLASSES

DATASET_INFO = {
    "mnist_modulo": {
        "concept_type": "binary",
        "concept_dim": 5,
        "num_classes": 10,
    },
    "pitfalls_mnist_without_45": {
        "concept_type": "binary",
        "concept_dim": 2,
        "num_classes": 2,
    },
    "pitfalls_random_concepts": {
        "concept_type": "binary",
        "concept_dim": 100,
        "num_classes": 2,
    },
    "pitfalls_synthetic": {
        "concept_type": "binary",
        "concept_dim": 6,
        "num_classes": 256,
    },
    "pitfalls_mnist_123456": {
        "concept_type": "binary",
        "concept_dim": 3,
        "num_classes": 2,
    },
    "cifar100": {
        "concept_type": "binary",
        "concept_dim": 20,
        "num_classes": 100,
        "class_names": list(SUPERCLASSES.keys()),
    },
    "cub": {"concept_type": "binary", "concept_dim": 112, "num_classes": 200},
    "oai": {"concept_type": "continuous", "concept_dim": 10, "num_classes": 4},
    "oai_binary": {"concept_type": "binary", "concept_dim": 40, "num_classes": 4},
    "imagenet": {"concept_type": "binary", "concept_dim": 65, "num_classes": 1000},
    "celeba": {"concept_type": "binary", "concept_dim": 6, "num_classes": 256},
    "aa2": {"concept_type": "binary", "concept_dim": 6, "num_classes": 50},
    "mimic_cxr": {
        "cardiomegaly": {"concept_type": "binary", "concept_dim": 60, "num_classes": 2},
        "effusion": {"concept_type": "binary", "concept_dim": 90, "num_classes": 2},
        "edema": {"concept_type": "binary", "concept_dim": 62, "num_classes": 2},
        "pneumonia": {"concept_type": "binary", "concept_dim": 97, "num_classes": 2},
        "pneumothorax": {"concept_type": "binary", "concept_dim": 35, "num_classes": 2},
        "cardiomegaly_transfer": {
            "concept_type": "binary",
            "concept_dim": 60,
            "num_classes": 2,
        },
    },
}


@functools.cache
def get_datasets(
    dataset_name: str,
    data_dir: str,
    resize_oai: bool = True,
    num_concepts: int = -1,
    backbone: str = "resnet34",
    subset="cardiomegaly",
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Get train, validation, and test splits for the given dataset.

    Parameters
    ----------
    name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)

    Returns
    -------
    train_dataset : Dataset
        Train dataset
    val_dataset : Dataset
        Validation dataset
    test_dataset : Dataset
        Test dataset
    """
    train_dataset, val_dataset, test_dataset = None, None, None

    if dataset_name == "mnist_modulo":
        train_dataset = MNISTModulo(root=data_dir, train=True)
        test_dataset = MNISTModulo(root=data_dir, train=False)

    elif dataset_name == "pitfalls_mnist_without_45":
        train_dataset = MNIST_45(root=data_dir, train=True)
        test_dataset = MNIST_45(root=data_dir, train=False)

    elif dataset_name == "pitfalls_random_concepts":
        train_dataset = DatasetC(root=data_dir, num_concepts=100, train=True)
        test_dataset = DatasetC(root=data_dir, num_concepts=100, train=False)

    elif dataset_name == "pitfalls_synthetic":
        train_dataset = DatasetD(train=True, num_concepts=num_concepts)
        val_dataset = DatasetD(train=False, num_concepts=num_concepts)
        test_dataset = DatasetD(train=False, num_concepts=num_concepts)

    elif dataset_name == "pitfalls_mnist_123456":
        train_dataset = DatasetE(root=data_dir, train=True)
        test_dataset = DatasetE(root=data_dir, train=False)

    elif dataset_name == "cifar100":
        if backbone == "vit_b_16":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224
                    ),  # Resizes to 224x224 with random cropping
                    transforms.RandomHorizontalFlip(),  # Random horizontal flipping
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),  # Optional: Color augmentation
                    transforms.ToTensor(),  # Converts image to tensor
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # Normalization for ImageNet
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(256),  # Resize shorter side to 256 pixels
                    transforms.CenterCrop(224),  # Center crop to 224x224
                    transforms.ToTensor(),  # Converts image to tensor
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # Normalization for ImageNet
                ]
            )

        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
                ]
            )
        train_dataset = CIFAR100(
            root=data_dir, train=True, transform=transform_train, download=True
        )
        test_dataset = CIFAR100(
            root=data_dir, train=False, transform=transform_test, download=True
        )

    elif dataset_name == "mimic_cxr":
        if backbone == "vit_b_16":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224
                    ),  # Resizes to 224x224 with random cropping
                    transforms.RandomHorizontalFlip(),  # Random horizontal flipping
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),  # Optional: Color augmentation
                    transforms.ToTensor(),  # Converts image to tensor
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # Normalization for ImageNet
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(256),  # Resize shorter side to 256 pixels
                    transforms.CenterCrop(224),  # Center crop to 224x224
                    transforms.ToTensor(),  # Converts image to tensor
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # Normalization for ImageNet
                ]
            )
        # elif backbone == "densenet121":
        #     class ToNumpyArray:
        #         def __call__(self, pic):
        #             return np.array(pic)
        #     transform_train = transforms.Compose([
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         #transforms.ToTensor(),
        #         ToNumpyArray(),
        #         transforms.Lambda(lambda img: xrv.datasets.normalize(img, 255)),  # Normalize to [-1024, 1024]
        #         transforms.Lambda(lambda img: img.mean(2)[None, ...]),           # Convert to single color channel
        #         xrv.datasets.XRayCenterCrop(),                                   # Center crop
        #         xrv.datasets.XRayResizer(224),                                   # Resize to 224x224                                           # Convert to PyTorch tensor
        #     ])
        #     transform_test = transforms.Compose([
        #         #transforms.ToTensor(),
        #         ToNumpyArray(),
        #         transforms.Lambda(lambda img: xrv.datasets.normalize(img, 255)),  # Normalize to [-1024, 1024]
        #         transforms.Lambda(lambda img: img.mean(2)[None, ...]),           # Convert to single color channel
        #         xrv.datasets.XRayCenterCrop(),                                   # Center crop
        #         xrv.datasets.XRayResizer(224),                                   # Resize to 224x224                                       # Convert to PyTorch tensor

        #     ])
        #     #transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(256),  # Resize shorter side to 256 pixels
                    transforms.CenterCrop(224),  # Center crop to 224x224
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(256),  # Resize shorter side to 256 pixels
                    transforms.CenterCrop(224),  # Center crop to 224x224
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
                ]
            )
        if subset == "cardiomegaly_transfer":
            train_dataset = CheX_Dataset(split="train", transform=transform_train)

            test_dataset = CheX_Dataset(split="val", transform=transform_test)

            val_dataset = CheX_Dataset(split="val", transform=transform_test)
        else:
            train_dataset = MIMIC_CXR(
                root=data_dir, subset=subset, split="train", transform=transform_train
            )

            test_dataset = MIMIC_CXR(
                root=data_dir, subset=subset, split="test", transform=transform_test
            )

            val_dataset = MIMIC_CXR(
                root=data_dir, subset=subset, split="val", transform=transform_test
            )

    elif dataset_name == "cub":
        transform_train = transforms.Compose(
            [
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )
        train_dataset = CUB(
            root=data_dir, split="train", transform=transform_train, download=True
        )
        val_dataset = CUB(
            root=data_dir, split="val", transform=transform_test, download=True
        )
        test_dataset = CUB(
            root=data_dir, split="test", transform=transform_test, download=True
        )

    elif dataset_name.startswith("oai"):
        transform_train = transforms.Compose(
            [
                transforms.Resize(224, antialias=False) if resize_oai else lambda x: x,
                transforms.Normalize(mean=-31334.48612, std=1324.959356),
                RandomTranslation(0.1, 0.1),
                lambda x: x.expand(3, -1, -1),  # expand to 3 channels
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224, antialias=False) if resize_oai else lambda x: x,
                transforms.Normalize(mean=-31334.48612, std=1324.959356),
                lambda x: x.expand(3, -1, -1),  # expand to 3 channels
            ]
        )
        train_dataset = OAI(
            root=data_dir,
            split="train",
            transform=transform_train,
            num_concepts=num_concepts,
            use_binary_concepts=(dataset_name == "oai_binary"),
        )
        val_dataset = OAI(
            root=data_dir,
            split="val",
            transform=transform_test,
            num_concepts=num_concepts,
            use_binary_concepts=(dataset_name == "oai_binary"),
        )
        test_dataset = OAI(
            root=data_dir,
            split="test",
            transform=transform_test,
            num_concepts=num_concepts,
            use_binary_concepts=(dataset_name == "oai_binary"),
        )

    elif dataset_name == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = ImageNet(
            data_dir / "train/",
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        test_dataset = ImageNet(
            data_dir / "val/",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

    elif dataset_name == "celeba":
        dataset_config = {
            "dataset": "celeba",
            "root_dir": data_dir,
            "image_size": 64,
            "num_classes": 1000,
            "use_imbalance": True,
            "use_binary_vector_class": True,
            "num_concepts": 6 if num_concepts == -1 else num_concepts,
            "label_binary_width": 1,
            "label_dataset_subsample": 12,
            "num_hidden_concepts": 2 if num_concepts == -1 else 0,
            "selected_concepts": False,
            "num_workers": 8,
            "sampling_percent": 1,
            "test_subsampling": 1,
            "backbone": backbone,
        }
        assert dataset_config['num_hidden_concepts'] == 2, dataset_config['num_hidden_concepts']
        assert dataset_config['num_concepts'] == 6, dataset_config['num_concepts']
        train_dataset, test_dataset, val_dataset = celeba_generate_data(
            dataset_config, dataset_config["root_dir"], split="all"
        )

    elif dataset_name == "aa2":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(
                    256
                ),  # Resize image so its shortest side is 256 pixels
                transforms.CenterCrop(64),
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = AA2(
            root=data_dir,
            split="train",
            transform=transform_train,
            download=True,
        )
        test_dataset = AA2(
            root=data_dir,
            split="val",
            transform=transform_test,
            download=True,
        )

    else:
        raise ValueError(f"Invalid dataset name:", dataset_name)

    # Create validation set if necessary
    if val_dataset is None:
        N = len(train_dataset)
        train_dataset, val_dataset = random_split(
            train_dataset,
            lengths=[N - int(0.15 * N), int(0.15 * N)],
            generator=torch.Generator().manual_seed(0),
        )

    return train_dataset, val_dataset, test_dataset


def get_datamodule(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    resize_oai: bool = True,
    num_concepts: int = -1,
    backbone: str = "resnet34",
    subset="cardiomegaly",
) -> pl.LightningDataModule:
    """
    Get a LightningDataModule for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for the data loaders
    """
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset_name,
        data_dir,
        resize_oai,
        num_concepts,
        backbone=backbone,
        subset=subset,
    )

    return pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # TODO: fix Ray Tune pickling error when num_workers > 0
    )


@functools.cache
def get_dummy_batch(
    dataset_name: str, data_dir: str, num_concepts: int, backbone: str, subset: str
) -> tuple[Any, Any]:
    """
    Get dummy batch for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    """
    loader = get_datamodule(
        dataset_name,
        data_dir,
        num_concepts=num_concepts,
        backbone=backbone,
        subset=subset,
    ).train_dataloader()
    return next(iter(loader))


@functools.cache
def get_concept_loss_fn(
    dataset_name: str, data_dir: str, num_concepts: int, backbone: str, subset: str
) -> nn.BCEWithLogitsLoss:
    """
    Get BCE concept loss function for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    """
    if dataset_name == "oai":
        # Get weighted mean squared error loss
        def weighted_mse(input, target):
            loss = (input - target) ** 2
            loss *= target.not_nan
            loss *= target.loss_class_wts
            return loss.mean()

        return weighted_mse

    else:
        if dataset_name == "mimic_cxr":
            subset_name = f"_{subset}"
        else:
            subset_name = ""
        weights_path = os.path.join(
            data_dir, f"{dataset_name}{subset_name}_pos_weights.pt"
        )

        if os.path.exists(weights_path):
            # Load the pos_weight tensor if it already exists
            pos_weight = torch.load(weights_path)
            print("Loaded weights from file.")
        else:
            # Get weighted binary cross entropy loss
            train_loader = get_datamodule(
                dataset_name,
                data_dir,
                num_workers=8,
                num_concepts=num_concepts,
                backbone=backbone,
                subset=subset,
            ).train_dataloader()
            if dataset_name == "mimic_cxr":
                concept_dim = DATASET_INFO[dataset_name][subset]["concept_dim"]
            else:
                concept_dim = DATASET_INFO[dataset_name]["concept_dim"]
            concepts_pos_count = torch.zeros(concept_dim)
            concepts_neg_count = torch.zeros(concept_dim)
            for (data, concepts), targets in tqdm(train_loader):
                concepts_pos_count += concepts.sum(dim=0)
                concepts_neg_count += (1 - concepts).sum(dim=0)

            pos_weight = concepts_neg_count / (concepts_pos_count + 1e-6)
            torch.save(pos_weight, weights_path)

        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@functools.cache
def get_target_loss_weights(
    dataset_name: str, data_dir: str, num_concepts: int, backbone: str, subset: str
) -> nn.BCEWithLogitsLoss:
    """
    Get BCE concept loss function for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir : str
        Directory where data is stored (or will be downloaded to)
    """
    if dataset_name == "oai":
        # Get weighted mean squared error loss
        def weighted_mse(input, target):
            loss = (input - target) ** 2
            loss *= target.not_nan
            loss *= target.loss_class_wts
            return loss.mean()

        return weighted_mse

    else:
        if dataset_name == "mimic_cxr":
            subset_name = f"_{subset}"
        else:
            subset_name = ""
        weights_path = os.path.join(
            data_dir, f"{dataset_name}{subset_name}_class_weights.pt"
        )

        if os.path.exists(weights_path):
            # Load the class weight tensor if it already exists
            class_weights = torch.load(weights_path)
            print("Loaded class weights from file.")
        else:
            # Get data loader
            train_loader = get_datamodule(
                dataset_name,
                data_dir,
                num_workers=16,
                num_concepts=num_concepts,
                backbone=backbone,
                subset=subset,
            ).train_dataloader()

            # Initialize counts for each class
            class_counts = torch.zeros(2)

            # Count occurrences of each class in the dataset
            for (data, concepts), targets in tqdm(train_loader):
                for target in targets:
                    class_counts[target] += 1

            # Calculate weights as the inverse of class frequencies
            total_samples = class_counts.sum()
            class_weights = total_samples / (class_counts + 1e-6)
            # Save the computed weights
            torch.save(class_weights, weights_path)

        return nn.CrossEntropyLoss(weight=class_weights)


class RandomTranslation:
    """
    Random translation transform.
    """

    def __init__(self, max_dx: float, max_dy: float, seed: int = 0):
        """
        Parameters
        ----------
        max_dx : float in interval [0, 1]
            Maximum absolute fraction for horizontal translation
        max_dy : float in interval [0, 1]
            Maximum absolute fraction for vertical translation
        seed : int
            Seed for the random number generator
        """
        self.max_dx, self.max_dy = max_dx, max_dy
        self.random = np.random.default_rng(seed)

    def __call__(self, img: Tensor) -> Tensor:
        dx = int(self.max_dx * img.shape[-2] * self.random.uniform(-1, -1))
        dy = int(self.max_dy * img.shape[-1] * self.random.uniform(-1, -1))
        return torch.roll(img, shifts=(dx, dy), dims=(-2, -1))
