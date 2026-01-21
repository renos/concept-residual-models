"""
MINE (Mutual Information Neural Estimation) implementation.

Based on "Mutual Information Neural Estimation" (Belghazi et al., 2018).
Provides neural network-based estimation of mutual information between
concepts and residuals for disentanglement.
"""

import torch
import torch.nn as nn
import math
from torch import Tensor


class EMALoss(torch.autograd.Function):
    """
    Custom autograd function for exponential moving average loss calculation.
    """

    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output * input.exp().detach() / (running_mean + 1e-6) / input.shape[0]
        )
        return grad, None


def ema(mu, alpha, past_ema):
    """Compute exponential moving average."""
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    """Compute EMA loss with running mean update."""
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean


class MINE(nn.Module):
    """
    Mutual Information Neural Estimator for mutual information.
    This class is adapted from the provided implementation.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_size: int = 400,
        loss_type: str = "mine",
        alpha: float = 0.01,
    ):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of X samples
        y_dim : int
            Dimension of Y samples
        hidden_size : int, default=400
            Dimension of hidden layers in the neural network
        loss_type : str, default='mine'
            Type of loss function to use ('mine', 'mine_biased', or 'fdiv')
        alpha : float, default=0.01
            EMA update parameter for the moving average
        """
        super().__init__()

        # Create a neural network for the mutual information estimator
        self.T = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.running_mean = 0
        self.loss_type = loss_type
        self.alpha = alpha

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the value of the MINE estimator for mutual information between X and Y.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples

        Returns
        -------
        torch.Tensor
            The estimated mutual information
        """
        # Make sure everything is on the right device
        if x.device != next(self.T.parameters()).device:
            self.T = self.T.to(x.device)

        # Create shuffled version of y for negative samples
        y_shuffled = y[torch.randperm(x.shape[0])]

        # Concatenate x and y for joint distribution
        xy = torch.cat([x, y], dim=1)
        x_y_shuffled = torch.cat([x, y_shuffled], dim=1)

        # T network outputs
        t_joint = self.T(xy).mean()
        t_marginal = self.T(x_y_shuffled)

        # Calculate the loss based on the chosen method
        if self.loss_type == "mine":
            second_term, self.running_mean = ema_loss(
                t_marginal, self.running_mean, self.alpha
            )
        elif self.loss_type == "fdiv":
            second_term = torch.exp(t_marginal - 1).mean()
        elif self.loss_type == "mine_biased":
            second_term = torch.logsumexp(t_marginal, 0) - math.log(t_marginal.shape[0])
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # The negative of the loss is the MI estimate
        mi_estimate = t_joint - second_term

        # Return negative MI estimate as loss
        return -mi_estimate

    def mi(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the mutual information estimate between X and Y.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples

        Returns
        -------
        torch.Tensor
            The estimated mutual information
        """
        # Convert numpy arrays to tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        with torch.no_grad():
            # Negative of loss is the MI estimate
            mi_estimate = -self.forward(x, y)

        return mi_estimate

    def learning_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the learning loss for MINE.
        This method is added to maintain API compatibility with CLUB.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples

        Returns
        -------
        torch.Tensor
            The learning loss
        """
        return self.forward(x, y)
