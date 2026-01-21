import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any

from .base import ConceptModel, ConceptLightningModel
from lib.club import CLUB, CLUBSample
from lib.mine import MINE


class MutualInformationLoss(nn.Module):
    """
    Creates a criterion that estimates an upper bound on the mutual information
    between x and y samples.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        mi_type="club",
    ):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of x samples
        y_dim : int
            Dimension of y samples
        hidden_dim : int
            Dimension of hidden layers in mutual information estimator
        lr : float
            Learning rate for mutual information estimator optimizer
        """
        super().__init__()
        if mi_type == "club":
            self.mi_estimator = CLUB(x_dim, y_dim, hidden_dim)
        elif mi_type == "sample":
            self.mi_estimator = CLUBSample(x_dim, y_dim, hidden_dim)
        elif mi_type == "mine":
            self.mi_estimator = MINE(x_dim, y_dim, hidden_size=hidden_dim)

        self.mi_optimizer = torch.optim.RMSprop(self.mi_estimator.parameters(), lr=lr)

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Estimate (an upper bound on) the mutual information for a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        return F.softplus(self.mi_estimator.forward(x, y))

    def step(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Run a single training step for the mutual information estimator
        on a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        # Unfreeze all params for MI estimator training
        self.train()
        for param in self.parameters():
            param.requires_grad = True

        # Train the MI estimator
        self.mi_optimizer.zero_grad()
        estimation_loss = self.mi_estimator.learning_loss(x.detach(), y.detach())
        estimation_loss.backward()
        self.mi_optimizer.step()

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        return estimation_loss


class MultiMutualInformationLoss(nn.Module):
    """
    Creates a criterion that estimates an upper bound on the mutual information
    between x and y samples.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64, lr: float = 1e-3):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of x samples
        y_dim : int
            Dimension of y samples
        hidden_dim : int
            Dimension of hidden layers in mutual information estimator
        lr : float
            Learning rate for mutual information estimator optimizer
        """
        super().__init__()
        mi_estimators = torch.nn.ModuleList()
        for i in range(y_dim):
            mi_estimators.append(CLUB(x_dim * (y_dim - 1) * 2, 1, hidden_dim))
        self.mi_estimators = mi_estimators
        self.mi_optimizer = torch.optim.RMSprop(self.mi_estimators.parameters(), lr=lr)

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Estimate (an upper bound on) the mutual information for a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        batch_size, num_concepts, embed_dim = x.shape
        probs = torch.tensor(0.0, device=x.device)

        for i in range(num_concepts):
            # Exclude the i-th concept
            x_except_i = torch.cat([x[:, :i, :], x[:, i + 1 :, :]], dim=1)
            # Reshape to (batch_size, -1)
            x_except_i = x_except_i.view(batch_size, -1)
            mi_estimate = F.softplus(
                self.mi_estimators[i](x_except_i, y[:, i].unsqueeze(1))
            )
            probs += mi_estimate

        return probs

    def step(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Run a single training step for the mutual information estimator
        on a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        # Unfreeze all params for MI estimator training
        self.train()
        for param in self.parameters():
            param.requires_grad = True

        # Train the MI estimator
        self.mi_optimizer.zero_grad()
        batch_size, num_concepts, embed_dim = x.shape
        estimation_loss = torch.tensor(0.0, device=x.device)

        for i in range(num_concepts):
            # Exclude the i-th concept
            x_except_i = torch.cat([x[:, :i, :], x[:, i + 1 :, :]], dim=1)
            # Reshape to (batch_size, -1)
            x_except_i = x_except_i.view(batch_size, -1)
            mi_loss_i = self.mi_estimators[i].learning_loss(
                x_except_i.detach(), y[:, i].unsqueeze(1).detach()
            )
            estimation_loss += mi_loss_i

        estimation_loss.backward()
        self.mi_optimizer.step()

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        return estimation_loss


class MutualInfoConceptLightningModel(ConceptLightningModel):
    """
    Concept model that minimizes the mutual information between
    concepts and residual.
    """

    def __init__(self, concept_model: ConceptModel, concept_embedding=False, **kwargs):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        concept_dim : int
            Dimension of concept vector
        residual_dim : int
            Dimension of residual vector
        mi_estimator_hidden_dim : int
            Dimension of hidden layer in mutual information estimator
        mi_optimizer_lr : float
            Learning rate for mutual information estimator optimizer
        """
        if concept_embedding:
            residual_loss_fn = MultiMutualInformationLoss(
                kwargs["residual_dim"],
                kwargs["concept_dim"],
                hidden_dim=kwargs["mi_estimator_hidden_dim"],
                lr=kwargs["mi_optimizer_lr"],
            )
        else:
            residual_loss_fn = MutualInformationLoss(
                kwargs["residual_dim"],
                kwargs["concept_dim"],
                hidden_dim=kwargs["mi_estimator_hidden_dim"],
                lr=kwargs["mi_optimizer_lr"],
                mi_type=kwargs.get("mi_type", "club"),
            )
        super().__init__(concept_model, residual_loss_fn=residual_loss_fn, **kwargs)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """
        Run one training step for the mutual information estimator.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        if isinstance(self.residual_loss_fn, MutualInformationLoss) or isinstance(
            self.residual_loss_fn, MultiMutualInformationLoss
        ):
            # Get concepts and residual
            with torch.no_grad():
                (data, concepts), targets = batch
                concept_logits, residual, target_logits = self(data, concepts=concepts)
            if type(concept_logits) == tuple:
                concept_logits = concept_logits[0]
            if type(residual) == tuple:
                residual = residual[0]

            # Calculate mutual information estimator loss
            mi_estimator_loss = self.residual_loss_fn.step(residual, concepts)
            mi_estimate = self.residual_loss_fn(residual, concepts)
            self.log("mi_estimator_loss", mi_estimator_loss, **self.log_kwargs)
            self.log("mi_estimate", mi_estimate.mean(), **self.log_kwargs)
