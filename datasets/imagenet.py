from __future__ import annotations

import torch

from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import datasets
import json
import os

### Constants

# load imagenet_superclasses
script_dir = os.path.dirname(__file__)
json_file_name = "imagenet_superclasses.json"
json_file_path = os.path.join(script_dir, json_file_name)
with open(json_file_path) as f:
    superclasses_to_class = json.load(f)
# convert human readable to ints and reverse / sort
sorted_final_dict = {k: superclasses_to_class[k] for k in sorted(superclasses_to_class)}
numbered_dict = {i: v for i, (k, v) in enumerate(sorted_final_dict.items(), start=0)}
reversed_dict = {}
for original_key, list_of_values in numbered_dict.items():
    for value in list_of_values:
        reversed_dict[value] = original_key

SUPERCLASSES_FROM_LABEL = {k: reversed_dict[k] for k in sorted(reversed_dict)}
NUM_SUPERCLASSES = len(superclasses_to_class)

### Dataset


class ImageNet(Dataset):
    """
    CIFAR-100 dataset (see https://www.cs.toronto.edu/~kriz/cifar.html).

    Each sample includes an image and a one-hot vector indicating the
    superclass for the image (e.g. 0 -> "aquatic mammals" for an image of a dolphin).
    """

    def __init__(self, *args, **kwargs):
        self.dataset = datasets.ImageFolder(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return (image, self.get_concepts(label)), label

    def get_concepts(self, label: int):
        superclass_idx = SUPERCLASSES_FROM_LABEL[label]
        return one_hot(torch.tensor(superclass_idx), NUM_SUPERCLASSES).float()
