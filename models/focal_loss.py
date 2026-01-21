import numpy as np
import math


import torch.nn.functional as F
import torch

from ray import tune

"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


def calculate_ortho_loss(concept_fn_out, residual_fn=None):
    cfn_out = concept_fn_out.reshape(-1, concept_fn_out.shape[-1])
    if residual_fn is not None:
        residual_fn = residual_fn.reshape(-1, residual_fn.shape[-1])
    cfn_out = cfn_out - torch.mean(cfn_out, dim=0)
    if residual_fn == None:
        residual_fn = cfn_out
        was_none = True
    else:
        was_none = False

    residual_fn = residual_fn - torch.mean(residual_fn, dim=0)
    ortho_matrix = torch.matmul(
        torch.transpose(cfn_out, 0, 1),
        residual_fn,
    )
    ortho_matrix = torch.div(ortho_matrix, cfn_out.shape[0])

    concept_std = torch.std(cfn_out, dim=0, unbiased=False)
    residual_std = torch.std(residual_fn, dim=0, unbiased=False)
    division = torch.ger(concept_std, residual_std)
    ortho_matrix = torch.div(ortho_matrix, division)

    if was_none:
        ortho_matrix = ortho_matrix[
            ~torch.eye(ortho_matrix.shape[0], ortho_matrix.shape[1], dtype=bool)
        ]
    ortho_loss = torch.mean(torch.abs(ortho_matrix))
    return ortho_loss


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)

    weights = (1.0 - beta) / (effective_num + 1e-3)

    weights = weights / torch.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = weights.float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels_one_hot, pos_weight=weights
        )
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(
            input=pred, target=labels_one_hot, weight=weights
        )
    return cb_loss
