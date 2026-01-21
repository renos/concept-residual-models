"""
Neural network extension modules for concept models.

This module provides utility classes that extend PyTorch's nn.Module
for building flexible concept bottleneck architectures.
"""

import torch.nn as nn

from functools import partial
from inspect import signature, _ParameterKind
from torch import Tensor
from typing import Callable


class Apply(nn.Module):
    """
    Wraps a callable function as a PyTorch module.

    Useful for creating simple network layers from lambda functions,
    e.g., for slicing tensors to extract concept or residual components.

    Example:
        >>> concept_network = Apply(lambda x: x[..., :concept_dim])
        >>> residual_network = Apply(lambda x: x[..., concept_dim:])
    """

    def __init__(self, fn: Callable, **fn_kwargs):
        super().__init__()
        self.forward = partial(fn, **fn_kwargs)


class BatchNormNd(nn.Module):
    """
    Dimension-agnostic batch normalization.

    Automatically selects BatchNorm1d, BatchNorm2d, or BatchNorm3d
    based on the input tensor dimensionality. Useful when the input
    dimension is not known at model construction time.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.bn = None
        self.kwargs = kwargs

    def forward(self, input: Tensor) -> Tensor:
        if self.bn is None:
            if input.ndim == 4:
                self.bn = nn.BatchNorm2d(input.shape[1], **self.kwargs)
            elif input.ndim == 5:
                self.bn = nn.BatchNorm3d(input.shape[1], **self.kwargs)
            else:
                self.bn = nn.BatchNorm1d(input.shape[1], **self.kwargs)

        return self.bn(input)


class Chain(nn.Sequential):
    """
    Sequential module that passes additional arguments through all layers.

    Unlike nn.Sequential, Chain forwards *args and **kwargs to each
    submodule, enabling modules that require extra context (e.g.,
    intervention flags) to receive them.

    Also supports the '+' operator for composing modules:
        >>> combined = module1 + module2
    """

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input

    def __add__(module_1: nn.Module, module_2: nn.Module):
        if not isinstance(module_1, Chain):
            module_1 = Chain(module_1)
        if not isinstance(module_2, Chain):
            module_2 = Chain(module_2)

        return Chain(*module_1, *module_2)

    __radd__ = __add__


class VariableKwargs(nn.Module):
    """
    Wrapper that filters keyword arguments to match a module's signature.

    Allows calling a module with extra kwargs that it doesn't accept,
    by filtering to only pass the kwargs the module's forward() expects.
    This is useful in concept models where different subnetworks may
    accept different sets of optional arguments.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def __getattr__(self, name: str):
        if name == 'module':
            return super().__getattr__(name)
        else:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        parameters = signature(self.module.forward).parameters.values()
        # If module accepts **kwargs, pass everything through
        if any(p.kind == _ParameterKind.VAR_KEYWORD for p in parameters):
            module_kwargs = kwargs
        else:
            # Otherwise, filter to only accepted parameters
            module_kwargs = {
                key: value for key, value in kwargs.items()
                if key in signature(self.module.forward).parameters.keys()
            }

        return self.module(*args, **module_kwargs)


# Monkey-patch nn.Module to support '+' operator for chaining modules
nn.Module.__add__ = Chain.__add__
nn.Module.__radd__ = Chain.__radd__
