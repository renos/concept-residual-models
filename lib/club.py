"""
CLUB (Contrastive Log-ratio Upper Bound) implementation.

Based on "A Contrastive Log-ratio Upper Bound of Mutual Information" (Cheng et al., 2020).
Provides a variational upper bound on mutual information, used as a regularizer
to minimize MI between concepts and residuals in D-CRM.
"""

import torch
import torch.nn as nn
from torch import Tensor


class CLUB(nn.Module):
    """
    Contrastive log-ratio upper bound estimator for mutual information.
    This class is adapted from https://github.com/Linear95/CLUB/.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int = 64):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of X samples
        y_dim : int
            Dimension of Y samples
        hidden_size : int, default=64
            Dimension of hidden layers in the approximation network q(Y|X)
        """
        super().__init__()

        # Mean of q(Y|X)
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )

        # Log-variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh(),
        )
        # self.p_mu = nn.Sequential(
        #     nn.Linear(x_dim, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(hidden_size, y_dim),
        # )

        # # Log-variance of q(Y|X)
        # self.p_logvar = nn.Sequential(
        #     nn.Linear(x_dim, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(hidden_size, y_dim),
        #     nn.Tanh(),
        # )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the value of the CLUB estimator for mutual information between X and Y.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples
        """
        # Mean and log-variance of q(Y|X)
        if x.device != next(self.p_mu.parameters()).device:
            self.p_mu = self.p_mu.to(x.device)
            self.p_logvar = self.p_logvar.to(x.device)
        mu, logvar = self.p_mu(x), self.p_logvar(x)

        # Log of conditional probability of positive sample pairs
        positive = -0.5 * (mu - y) ** 2 / logvar.exp()

        # Log of conditional probability of negative sample pairs
        y = y.unsqueeze(0)  # shape (1, num_samples, y_dim)
        prediction = mu.unsqueeze(1)  # shape (num_samples, 1, y_dim)
        negative = -0.5 * ((prediction - y) ** 2).mean(dim=1) / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikelihood(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the (unnormalized) log-likelihood of the approximation q(Y|X)
        with the given samples.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples
        """
        # Mean and log-variance of q(Y|X)
        mu, logvar = self.p_mu(x), self.p_logvar(x)
        out = -((mu - y) ** 2) / logvar.exp() - logvar
        return out.sum(dim=1).mean(dim=0)

    def learning_loss(self, x: Tensor, y: Tensor):
        """
        Get the learning loss of the approximation q(Y|X) of the given samples.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples
        """
        return -self.loglikelihood(x, y)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )

        # Log-variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-((mu - y_samples) ** 2) / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = -((mu - y_samples) ** 2) / logvar.exp()
        negative = -((mu - y_samples[random_index]) ** 2) / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)
