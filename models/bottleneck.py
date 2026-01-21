from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from lib.iterative_normalization import IterNorm, IterNormRotation
from nn_extensions import BatchNormNd



def make_bottleneck_layer(
    bottleneck_dim: int,
    norm_type: str | None = None,
    affine: bool = True,
    T_whitening: int = 5,
    cw_activation_mode: str = 'mean',
    **kwargs) -> nn.Module:
    """
    Make bottleneck layer for a concept model.

    Parameters
    ----------
    bottleneck_dim : int
        Dimension of bottleneck layer (input and output)
    norm_type : str or None
        Type of normalization to apply to bottleneck layer
    affine : bool
        Whether to include a learnable affine transformation
    T_whitening : int
        Number of whitening iterations (for 'iter_norm' and 'concept_whitening')
    cw_activation_mode : one of {'mean', 'max', 'pos_mean', 'pool_max'}
        Mode for concept whitening activation
    """
    if norm_type == 'batch_norm':
        return BatchNormNd(affine=affine)
    elif norm_type == 'layer_norm':
        return nn.LayerNorm(bottleneck_dim)
    elif norm_type == 'instance_norm':
        return nn.InstanceNorm1d(bottleneck_dim, affine=affine)
    elif norm_type == 'spectral_norm':
        return nn.utils.spectral_norm(nn.LazyLinear(bottleneck_dim))
    elif norm_type == 'iter_norm':
        return IterNorm(
            bottleneck_dim,
            num_channels=bottleneck_dim,
            T=T_whitening,
            dim=2,
            momentum=1.0,
            affine=affine,
        )
    elif norm_type == 'concept_whitening':
        return ConceptWhitening(
            bottleneck_dim,
            num_channels=bottleneck_dim,
            T=T_whitening,
            momentum=1.0,
            affine=affine,
            activation_mode=cw_activation_mode,
        )
    elif norm_type is None:
        return nn.Identity()

    raise ValueError(f'Unknown norm_type: {norm_type}')


class ConceptWhitening(IterNormRotation):
    """
    Concept whitening layer (with support for arbitrary number of input dimensions).
    """

    def forward(self, x: Tensor):
        bottleneck = x
        while bottleneck.ndim < 4:
            bottleneck = bottleneck.unsqueeze(-1)
        if bottleneck.ndim > 4:
            bottleneck = bottleneck.view(-1, *bottleneck.shape[-3:])

        return super().forward(bottleneck).view(*x.shape)
