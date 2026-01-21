from __future__ import annotations

import torch

from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import datasets


### Constants

SUPERCLASSES = {
    "aquatic mammals": {"beaver", "dolphin", "otter", "seal", "whale"},
    "fish": {"aquarium_fish", "flatfish", "ray", "shark", "trout"},
    "flowers": {"orchid", "poppy", "rose", "sunflower", "tulip"},
    "food containers": {"bottle", "bowl", "can", "cup", "plate"},
    "fruit and vegetables": {"apple", "mushroom", "orange", "pear", "sweet_pepper"},
    "household electrical devices": {
        "clock",
        "keyboard",
        "lamp",
        "telephone",
        "television",
    },
    "household furniture": {"bed", "chair", "couch", "table", "wardrobe"},
    "insects": {"bee", "beetle", "butterfly", "caterpillar", "cockroach"},
    "large carnivores": {"bear", "leopard", "lion", "tiger", "wolf"},
    "large man-made outdoor things": {
        "bridge",
        "castle",
        "house",
        "road",
        "skyscraper",
    },
    "large natural outdoor scenes": {"cloud", "forest", "mountain", "plain", "sea"},
    "large omnivores and herbivores": {
        "camel",
        "cattle",
        "chimpanzee",
        "elephant",
        "kangaroo",
    },
    "medium-sized mammals": {"fox", "porcupine", "possum", "raccoon", "skunk"},
    "non-insect invertebrates": {"crab", "lobster", "snail", "spider", "worm"},
    "people": {"baby", "boy", "girl", "man", "woman"},
    "reptiles": {"crocodile", "dinosaur", "lizard", "snake", "turtle"},
    "small mammals": {"hamster", "mouse", "rabbit", "shrew", "squirrel"},
    "trees": {"maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"},
    "vehicles 1": {"bicycle", "bus", "motorcycle", "pickup_truck", "train"},
    "vehicles 2": {"lawn_mower", "rocket", "streetcar", "tank", "tractor"},
}
LABEL_TO_SUPERCLASS = {
    label: sorted(SUPERCLASSES.keys()).index(superclass)
    for superclass, labels in SUPERCLASSES.items()
    for label in labels
}
CLASSES = [label for labels in SUPERCLASSES.values() for label in labels]


def map_indices(a, b):
    return {i: b.index(item) for i, item in enumerate(a)}


### Dataset
class CIFAR100(Dataset):
    """
    CIFAR-100 dataset (see https://www.cs.toronto.edu/~kriz/cifar.html).

    Each sample includes an image and a one-hot vector indicating the
    superclass for the image (e.g. 0 -> "aquatic mammals" for an image of a dolphin).
    """

    def __init__(self, *args, **kwargs):
        self.dataset = datasets.CIFAR100(*args, **kwargs)
        self.remap = map_indices(self.dataset.classes, CLASSES)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        remapped_label = self.remap[label]
        return (image, self.get_concepts(remapped_label)), label

    def get_concepts(self, label: int):
        superclass_idx = LABEL_TO_SUPERCLASS[CLASSES[label]]
        return one_hot(torch.tensor(superclass_idx), len(SUPERCLASSES)).float()
