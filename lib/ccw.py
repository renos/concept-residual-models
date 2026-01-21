"""
EYE (Expert Yielded Estimation) implementation.

Provides attribution-based concept weighting for interpretability,
based on gradient-based feature attribution methods.
"""

import torch
from torch.autograd import grad


def EYE(r, x):
    """
    expert yielded estimation
    r: risk factors indicator (d,)
    x: attribution (d,)

    """
    assert r.shape[-1] == x.shape[-1]  # can broadcast
    l1 = (x * (1 - r)).abs().sum(-1)
    l2sq = ((r * x) ** 2).sum(-1)
    return l1 + torch.sqrt(l1**2 + l2sq)
