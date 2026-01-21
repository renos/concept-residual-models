"""
Utility functions for concept residual models.

This module provides helper functions for:
- Metrics computation (accuracy, AUROC)
- Neural network construction (MLPs, CNNs)
- Concept embedding model creation
- Cross-correlation computation
- CUDA device management
- Ray Tune configuration processing
"""

from __future__ import annotations

import os
import pynvml
import torch
import torch.nn as nn

from collections import ChainMap
from ray.tune.search.variant_generator import generate_variants, grid_search
from torch import Tensor
from torchmetrics import Accuracy, AUROC
from typing import Any
from open_clip import create_model_from_pretrained
from transformers import ViTForImageClassification


def zero_loss_fn(*tensors: Tensor) -> Tensor:
    """
    Dummy loss function that returns zero.

    Useful as a placeholder when a loss component should be disabled.

    Parameters
    ----------
    tensors : Tensor
        Any tensors (used only to determine the device)

    Returns
    -------
    Tensor
        Zero tensor on the same device as the input
    """
    return torch.tensor(0.0, device=tensors[0].device)


def accuracy(logits: Tensor, targets: Tensor, task: str = "multiclass") -> Tensor:
    """
    Compute accuracy from logits and targets.

    Parameters
    ----------
    logits : Tensor
        Model output logits
    targets : Tensor
        Ground truth labels
    task : str, optional
        Task type: "binary" or "multiclass". Defaults to "multiclass".

    Returns
    -------
    Tensor
        Accuracy score
    """
    if task == "binary":
        preds = logits.sigmoid()
        accuracy_fn = Accuracy(task="binary").to(logits.device)
    else:
        preds = logits.argmax(dim=-1)
        accuracy_fn = Accuracy(task="multiclass", num_classes=logits.shape[-1]).to(
            logits.device
        )

    return accuracy_fn(preds, targets)


def auroc(logits: Tensor, targets: Tensor, task: str = "multiclass") -> Tensor:
    """
    Compute AUROC from logits and targets.

    Parameters
    ----------
    logits : Tensor
        Model output logits
    targets : Tensor
        Ground truth labels
    task : str, optional
        Task type: "binary" or "multiclass". Defaults to "multiclass".

    Returns
    -------
    Tensor
        AUROC score
    """
    if task == "binary":
        preds = logits.sigmoid()
        auroc_fn = AUROC(task="binary").to(logits.device)
    else:
        preds = logits.softmax(dim=-1)
        auroc_fn = AUROC(task="multiclass", num_classes=logits.shape[-1]).to(
            logits.device
        )

    return auroc_fn(preds, targets)


def to_device(
    data: Tensor | tuple[Tensor] | list[Tensor], device: torch.device | str
) -> Tensor | tuple[Tensor] | list[Tensor]:
    """
    Recursively move a tensor or collection of tensors to the given device.

    Parameters
    ----------
    data : Tensor or tuple[Tensor] or list[Tensor]
        Tensor or nested collection of tensors
    device : torch.device or str
        Target device

    Returns
    -------
    Tensor or tuple[Tensor] or list[Tensor]
        Data moved to the specified device, preserving structure

    Raises
    ------
    ValueError
        If data type is not supported
    """
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]

    raise ValueError(f"Unsupported data type: {type(data)}")


def unwrap(model: nn.Module) -> nn.Module:
    """
    Unwrap a model from DataParallel or DistributedDataParallel wrapper.

    Parameters
    ----------
    model : nn.Module
        Potentially wrapped model

    Returns
    -------
    nn.Module
        Unwrapped model (or original if not wrapped)
    """
    return getattr(model, "module", model)


def make_mlp(
    output_dim: int,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    flatten_input: bool = False,
    add_layer_norm: bool = False,
) -> nn.Module:
    """
    Create a multi-layer perceptron with lazy input dimension inference.

    Parameters
    ----------
    output_dim : int
        Output dimension
    hidden_dim : int, optional
        Hidden layer dimension. Defaults to 256.
    num_hidden_layers : int, optional
        Number of hidden layers. Defaults to 2.
    flatten_input : bool, optional
        Whether to flatten input. Defaults to False.
    add_layer_norm : bool, optional
        Whether to add layer normalization after each hidden layer.
        Defaults to False.

    Returns
    -------
    nn.Module
        MLP network
    """
    hidden_layers = []
    for _ in range(num_hidden_layers):
        hidden_layers.append(nn.LazyLinear(hidden_dim))
        hidden_layers.append(nn.ReLU())
        if add_layer_norm:
            hidden_layers.append(nn.LayerNorm(hidden_dim))

    pre_input_layer = nn.Flatten() if flatten_input else nn.Identity()
    return nn.Sequential(pre_input_layer, *hidden_layers, nn.LazyLinear(output_dim))


def make_cnn(
    output_dim: int, cnn_type: str = "resnet18", load_weights: bool = True
) -> nn.Module:
    """
    Create a convolutional neural network backbone with a custom output head.

    Supports multiple architectures with optional ImageNet pretrained weights.

    Parameters
    ----------
    output_dim : int
        Output dimension (number of classes or bottleneck size)
    cnn_type : str, optional
        CNN architecture. One of: "resnet18", "resnet34", "densenet121",
        "vit_b_16", "inception_v3", "dfn2b_clip_vit_b_16". Defaults to "resnet18".
    load_weights : bool, optional
        Whether to load pretrained ImageNet weights. Defaults to True.

    Returns
    -------
    nn.Module
        CNN model with modified final layer

    Raises
    ------
    ValueError
        If cnn_type is not recognized
    """
    if cnn_type == "resnet18":
        from torchvision.models.resnet import resnet18, ResNet18_Weights

        model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if load_weights else None
        )
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    elif cnn_type == "resnet34":
        from torchvision.models.resnet import resnet34, ResNet34_Weights

        model = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1 if load_weights else None
        )
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    elif cnn_type == "densenet121":
        from torchvision.models import densenet121, DenseNet121_Weights

        model = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1 if load_weights else None
        )
        model.classifier = nn.Linear(model.classifier.in_features, output_dim)
        model = model.to(dtype=torch.bfloat16)
        return model

    elif cnn_type == "vit_b_16":
        model_name = "google/vit-base-patch16-224"
        model = ViTForImageClassification.from_pretrained(
            model_name, return_dict=True, torch_dtype=torch.bfloat16
        )
        model.classifier = nn.Linear(768, output_dim, bias=True)
        return model

    elif cnn_type == "inception_v3":
        from torchvision.models.inception import inception_v3, Inception_V3_Weights

        model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if load_weights else None
        )
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        model.aux_logits = False
        return model

    elif cnn_type == "dfn2b_clip_vit_b_16":
        model, _ = create_model_from_pretrained("hf-hub:apple/DFN2B-CLIP-ViT-B-16")
        if not load_weights:
            model.init_parameters()

        class Detach(nn.Module):
            """Detaches gradients from the input tensor."""

            def forward(self, x):
                return x.detach()

        new_model = nn.Sequential(
            model.visual,
            Detach(),
            nn.Linear(512, output_dim),
        )
        return new_model

    raise ValueError(f"Unknown CNN type: {cnn_type}")


def make_concept_embedding_model(
    in_dim: int,
    emb_size: int,
    n_concepts: int,
    embedding_activation: str = "leakyrelu",
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """
    Create concept embedding generators for Concept Embedding Models (CEM).

    Each concept has a probability generator and a context (embedding) generator.
    The context generator outputs 2*emb_size features (positive and negative
    embeddings that are later mixed based on concept probability).

    Parameters
    ----------
    in_dim : int
        Input dimension (from backbone)
    emb_size : int
        Size of each concept embedding
    n_concepts : int
        Number of concepts
    embedding_activation : str, optional
        Activation function: None, "sigmoid", "leakyrelu", or "relu".
        Defaults to "leakyrelu".

    Returns
    -------
    tuple[nn.ModuleList, nn.ModuleList]
        (concept_prob_generators, concept_context_generators)
    """
    concept_prob_generators = nn.ModuleList()
    concept_context_generators = nn.ModuleList()

    for _ in range(n_concepts):
        # Context generator: outputs positive and negative embedding portions
        if embedding_activation is None:
            context_gen = nn.Sequential(
                nn.Linear(in_dim, 2 * emb_size),
            )
        elif embedding_activation == "sigmoid":
            context_gen = nn.Sequential(
                nn.Linear(in_dim, 2 * emb_size),
                nn.Sigmoid(),
            )
        elif embedding_activation == "leakyrelu":
            context_gen = nn.Sequential(
                nn.Linear(in_dim, 2 * emb_size),
                nn.LeakyReLU(),
            )
        elif embedding_activation == "relu":
            context_gen = nn.Sequential(
                nn.Linear(in_dim, 2 * emb_size),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown embedding activation: {embedding_activation}")

        concept_context_generators.append(context_gen)

        # Probability generator: predicts concept probability from embedding
        concept_prob_generators.append(nn.Linear(2 * emb_size, 1))

    return concept_prob_generators, concept_context_generators


def cross_correlation(X: Tensor, Y: Tensor) -> Tensor:
    """
    Compute the cross-correlation matrix between X and Y.

    Both inputs are z-score normalized before computing the correlation.

    Parameters
    ----------
    X : Tensor of shape (batch_size, x_dim)
        First set of samples
    Y : Tensor of shape (batch_size, y_dim)
        Second set of samples

    Returns
    -------
    Tensor of shape (x_dim, y_dim)
        Cross-correlation matrix
    """
    eps = torch.tensor(1e-6)
    X = (X - X.mean(dim=0)) / torch.maximum(X.std(dim=0), eps)
    Y = (Y - Y.mean(dim=0)) / torch.maximum(Y.std(dim=0), eps)
    return torch.bmm(X.unsqueeze(-1), Y.unsqueeze(1)).mean(dim=0)


# =============================================================================
# CUDA Utilities
# =============================================================================


def set_cuda_visible_devices(available_memory_threshold: float) -> None:
    """
    Set CUDA_VISIBLE_DEVICES to GPUs with sufficient available memory.

    Useful when running processes with fractional GPUs to select devices
    that have the required memory available.

    Parameters
    ----------
    available_memory_threshold : float
        Threshold in range [0, 1] for fraction of available GPU memory.
        Only GPUs with at least this fraction of memory free will be visible.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError_LibraryNotFound:
        return

    available_devices = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if memory_info.free / memory_info.total >= available_memory_threshold:
            available_devices.append(i)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_devices))


# =============================================================================
# Ray Tune Utilities
# =============================================================================


def process_grid_search_tuples(config: dict) -> dict:
    """
    Process configuration dictionary with grid search over tuples.

    Allows specifying multiple related hyperparameters that should vary together
    using tuple keys and grid_search over tuple values.

    Parameters
    ----------
    config : dict
        Configuration dictionary with entries of the form:
        - k: v (single key-value pair)
        - (k_0, k_1, ..., k_n): grid_search([(v0_0, ..., v0_n), ...])
          (tuple key with grid search over corresponding tuple values)

    Returns
    -------
    dict
        Grid search configuration with expanded tuples

    Example
    -------
    >>> config = {
    ...     ('model_type', 'beta'): grid_search([
    ...         ('latent_residual', 0),
    ...         ('mi_residual', 1.0),
    ...     ]),
    ...     'lr': 1e-4,
    ... }
    This produces grid search over:
    - {'model_type': 'latent_residual', 'beta': 0, 'lr': 1e-4}
    - {'model_type': 'mi_residual', 'beta': 1.0, 'lr': 1e-4}
    """
    # Turn all keys into tuples, and all values into tuples
    config = {
        k if isinstance(k, tuple) else (k,): v if isinstance(k, tuple) else (v,)
        for k, v in config.items()
    }

    # Convert into a grid search over individual config dictionaries
    merge_dicts = lambda dicts: dict(ChainMap(*dicts))
    config = grid_search(
        [
            merge_dicts(dict(zip(k, v)) for k, v in reversed(spec.items()))
            for _, spec in generate_variants(config)
        ]
    )

    return config


def remove_prefix(state_dict: dict, prefix: str) -> dict:
    """
    Remove a prefix from all keys in a state dictionary.

    Useful for loading checkpoints saved with DataParallel or
    DistributedDataParallel wrappers.

    Parameters
    ----------
    state_dict : dict
        Model state dictionary
    prefix : str
        Prefix to remove (e.g., "module.")

    Returns
    -------
    dict
        State dictionary with prefix removed from matching keys
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def remove_keys_with_prefix(state_dict: dict, prefix: str) -> dict:
    """
    Remove all keys with a given prefix from a state dictionary.

    Useful for removing unwanted module weights when loading checkpoints.

    Parameters
    ----------
    state_dict : dict
        Model state dictionary
    prefix : str
        Prefix identifying keys to remove

    Returns
    -------
    dict
        State dictionary with matching keys removed
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            new_state_dict[key] = value
    return new_state_dict


class RayConfig(dict):
    """
    Dictionary subclass for handling Ray Tune configurations with grid search.

    Provides get/set methods that work correctly with nested grid_search
    configurations, allowing uniform access regardless of whether a value
    is a single value or part of a grid search.
    """

    def get(self, key: str, default: Any = ...) -> Any:
        """
        Get a value from the config, handling grid_search transparently.

        Parameters
        ----------
        key : str
            Configuration key
        default : Any, optional
            Default value if key is not found. If not provided, raises KeyError.

        Returns
        -------
        Any
            Configuration value

        Raises
        ------
        KeyError
            If key is not found and no default provided
        AssertionError
            If key has inconsistent values across grid search variants
        """
        try:
            if key in self:
                value = self[key]
                if (
                    isinstance(value, dict)
                    and len(value) == 1
                    and "grid_search" in value
                ):
                    return value["grid_search"]
                else:
                    return value

            elif "train_loop_config" in self:
                return self.get(self["train_loop_config"], key, default=default)

            elif "grid_search" in self:
                values = {item[key] for item in self["grid_search"]}
                assert len(values) == 1, f"Inconsistent values for {key}: {values}"
                return next(iter(values))

            raise KeyError

        except KeyError:
            if default is not ...:
                return default

        raise KeyError(f"Key not found: {key}")

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the config, propagating to all grid search variants.

        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Value to set
        """
        if "grid_search" in self:
            for item in self["grid_search"]:
                item[key] = value
        else:
            self[key] = value

    def update(self, other: dict) -> None:
        """
        Update the config from another dictionary.

        Parameters
        ----------
        other : dict
            Dictionary of key-value pairs to update
        """
        for key, value in other.items():
            self.set(key, value)
