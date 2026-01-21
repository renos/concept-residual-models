"""
Adversarial decorrelation for concept-residual disentanglement.

Uses gradient reversal to train a discriminator that predicts concepts from
residuals, effectively minimizing mutual information through adversarial learning.
"""

import torch
import torch.nn as nn
from torch import Tensor


class AdversarialDecorrelation(nn.Module):
    """
    Adversarial decorrelation for minimizing mutual information between concepts and residuals.

    This module trains a discriminator to predict concepts from residuals, then uses gradient
    reversal to make the residual encoder fool the discriminator, effectively minimizing I(C;R).

    The approach is similar to CLUB-based MI minimization but uses gradient reversal instead
    of alternating optimization.
    """

    def __init__(self, residual_dim: int, concept_dim: int, hidden_size: int = 64):
        """
        Parameters
        ----------
        residual_dim : int
            Dimension of residual representations
        concept_dim : int
            Dimension of concept representations
        hidden_size : int, default=64
            Dimension of hidden layers in the discriminator network
        """
        super().__init__()

        # Discriminator network: predicts concepts from residuals
        self.discriminator = nn.Sequential(
            nn.Linear(residual_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, concept_dim),
        )

    def forward(self, residual: Tensor, concepts: Tensor) -> Tensor:
        """
        Compute the discriminator loss (concept prediction from residuals).

        This loss is used to train the discriminator to predict concepts from residuals.
        When applied to the residual encoder with reversed gradients, it encourages
        decorrelation.

        Parameters
        ----------
        residual : torch.Tensor of shape (batch_size, residual_dim)
            Residual representations
        concepts : torch.Tensor of shape (batch_size, concept_dim)
            Concept representations (ground truth or predicted)

        Returns
        -------
        torch.Tensor
            MSE loss for concept prediction
        """
        # Ensure discriminator is on the same device as input
        if residual.device != next(self.discriminator.parameters()).device:
            self.discriminator = self.discriminator.to(residual.device)

        # Predict concepts from residuals
        concept_pred = self.discriminator(residual)

        # MSE loss for concept prediction
        loss = torch.mean((concept_pred - concepts) ** 2)

        return loss

    def discriminator_loss(self, residual: Tensor, concepts: Tensor) -> Tensor:
        """
        Compute discriminator loss for training the discriminator.

        This is the same as forward() but provides explicit naming for clarity.

        Parameters
        ----------
        residual : torch.Tensor of shape (batch_size, residual_dim)
            Residual representations
        concepts : torch.Tensor of shape (batch_size, concept_dim)
            Concept representations

        Returns
        -------
        torch.Tensor
            MSE loss for concept prediction
        """
        return self.forward(residual.detach(), concepts.detach())

    def decorrelation_loss(self, residual: Tensor, concepts: Tensor, lambda_adv: float = 1.0) -> Tensor:
        """
        Compute decorrelation loss for the residual encoder using gradient reversal.

        This loss encourages the residual encoder to produce representations that
        fool the discriminator, thereby decorrelating concepts from residuals.

        Parameters
        ----------
        residual : torch.Tensor of shape (batch_size, residual_dim)
            Residual representations (with gradients)
        concepts : torch.Tensor of shape (batch_size, concept_dim)
            Concept representations
        lambda_adv : float, default=1.0
            Strength of adversarial decorrelation

        Returns
        -------
        torch.Tensor
            Negative discriminator loss (for gradient reversal effect)
        """
        # Predict concepts from residuals
        concept_pred = self.discriminator(residual)
        print(concepts.shape)

        # MSE loss for concept prediction
        # Detach concepts since we don't want to backprop through concept network
        disc_loss = torch.mean((concept_pred - concepts.detach()) ** 2)

        # Return negative loss to create gradient reversal effect
        # When this is minimized, it maximizes the discriminator loss,
        # making residuals uninformative about concepts
        return -lambda_adv * disc_loss


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.

    Forward pass: identity function
    Backward pass: reverses and scales gradients
    """

    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


class AdversarialDecorrelationGRL(nn.Module):
    """
    Adversarial decorrelation using explicit Gradient Reversal Layer (GRL).

    This is an alternative implementation that uses an explicit gradient reversal layer
    instead of manual gradient reversal. Both approaches are equivalent but this one
    follows the classical GRL implementation from domain adaptation literature.
    """

    def __init__(self, residual_dim: int, concept_dim: int, hidden_size: int = 64, max_error: float = 1.0):
        """
        Parameters
        ----------
        residual_dim : int
            Dimension of residual representations
        concept_dim : int
            Dimension of concept representations
        hidden_size : int, default=64
            Dimension of hidden layers in the discriminator network
        max_error : float, default=1.0
            Maximum value to clamp MSE loss to prevent exploding gradients
        """
        super().__init__()

        self.max_error = max_error

        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(residual_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, concept_dim),
        )

    def forward(self, residual: Tensor, concepts: Tensor, lambda_adv: float = 1.0,
                use_grl: bool = False) -> Tensor:
        """
        Compute the adversarial decorrelation loss.

        Parameters
        ----------
        residual : torch.Tensor of shape (batch_size, residual_dim)
            Residual representations
        concepts : torch.Tensor of shape (batch_size, concept_dim)
            Concept representations
        lambda_adv : float, default=1.0
            Gradient reversal strength
        use_grl : bool, default=False
            Whether to use gradient reversal layer

        Returns
        -------
        torch.Tensor
            Concept prediction loss
        """
        # Ensure discriminator is on the same device as input
        if residual.device != next(self.discriminator.parameters()).device:
            self.discriminator = self.discriminator.to(residual.device)

        # Apply gradient reversal if requested
        if use_grl:
            residual_reversed = GradientReversalLayer.apply(residual, lambda_adv)
        else:
            residual_reversed = residual

        # Predict concepts from (possibly reversed) residuals
        concept_pred = self.discriminator(residual_reversed)

        # MSE loss with clamping to prevent unbounded gradients
        mse = (concept_pred - concepts) ** 2
        # Clamp to maximum value to prevent exploding gradients
        mse_clamped = torch.clamp(mse, max=self.max_error)
        loss = torch.mean(mse_clamped)

        return loss

    def discriminator_loss(self, residual: Tensor, concepts: Tensor) -> Tensor:
        """
        Compute discriminator loss (without gradient reversal).

        Parameters
        ----------
        residual : torch.Tensor of shape (batch_size, residual_dim)
            Residual representations
        concepts : torch.Tensor of shape (batch_size, concept_dim)
            Concept representations

        Returns
        -------
        torch.Tensor
            MSE loss for concept prediction
        """
        return self.forward(residual.detach(), concepts, lambda_adv=0.0, use_grl=False)
