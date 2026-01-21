import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any

from .base import ConceptModel, ConceptLightningModel
from lib.adversarial_decorrelation import AdversarialDecorrelation


class AdversarialDecorrelationLoss(nn.Module):
    """
    Creates a criterion that uses adversarial training to decorrelate
    concepts from residuals, minimizing mutual information I(C;R).
    """

    def __init__(
        self,
        residual_dim: int,
        concept_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        lambda_adv: float = 1.0,
        max_error: float = 1.0,
    ):
        """
        Parameters
        ----------
        residual_dim : int
            Dimension of residual samples
        concept_dim : int
            Dimension of concept samples
        hidden_dim : int
            Dimension of hidden layers in discriminator network
        lr : float
            Learning rate for discriminator optimizer
        lambda_adv : float
            Adversarial decorrelation strength
        max_error : float
            Maximum value to clamp MSE loss to prevent exploding gradients
        """
        super().__init__()
        # Use GRL-based implementation instead of negative loss
        from lib.adversarial_decorrelation import AdversarialDecorrelationGRL
        self.adv_decorr = AdversarialDecorrelationGRL(residual_dim, concept_dim, hidden_dim, max_error)
        self.discriminator_optimizer = torch.optim.Adam(
            self.adv_decorr.parameters(), lr=lr
        )
        self.lambda_adv = lambda_adv


        # Freeze all params for discriminator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, residual: Tensor, concepts: Tensor) -> Tensor:
        """
        Compute the decorrelation loss for the residual encoder.

        This uses gradient reversal layer to encourage decorrelation,
        making residuals uninformative about concepts.

        Parameters
        ----------
        residual : Tensor of shape (..., residual_dim)
            Batch of residual samples
        concepts : Tensor of shape (..., concept_dim)
            Batch of concept samples
        """
        # Use GRL for gradient reversal
        return self.adv_decorr(residual, concepts, lambda_adv=self.lambda_adv, use_grl=True)

    def step(self, residual: Tensor, concepts: Tensor) -> Tensor:
        """
        Run a single training step for the discriminator on a batch of samples.

        Parameters
        ----------
        residual : Tensor of shape (..., residual_dim)
            Batch of residual samples
        concepts : Tensor of shape (..., concept_dim)
            Batch of concept samples
        """
        # Unfreeze all params for discriminator training
        self.train()
        for param in self.parameters():
            param.requires_grad = True

        # Train the discriminator
        self.discriminator_optimizer.zero_grad()
        disc_loss = self.adv_decorr.discriminator_loss(residual.detach(), concepts.detach())
        disc_loss.backward()
        self.discriminator_optimizer.step()

        # Freeze all params for discriminator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        return disc_loss


class AdversarialDecorrelationConceptLightningModel(ConceptLightningModel):
    """
    Concept model that uses adversarial decorrelation to minimize
    mutual information between concepts and residual.
    """

    def __init__(self, concept_model: ConceptModel, **kwargs):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        residual_dim : int
            Dimension of residual vector
        concept_dim : int
            Dimension of concept vector
        adv_decorr_hidden_dim : int
            Dimension of hidden layer in discriminator
        adv_decorr_optimizer_lr : float
            Learning rate for discriminator optimizer
        lambda_adv : float
            Adversarial decorrelation strength
        """
        residual_loss_fn = AdversarialDecorrelationLoss(
            kwargs["residual_dim"],
            kwargs["concept_dim"],
            hidden_dim=kwargs.get("adv_decorr_hidden_dim", 256),
            lr=kwargs.get("adv_decorr_optimizer_lr", 1e-3),
            lambda_adv=kwargs.get("lambda_adv", 1.0),
            max_error=kwargs.get("max_error", 1.0),
        )
        super().__init__(concept_model, residual_loss_fn=residual_loss_fn, **kwargs)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """
        Run one training step for the discriminator.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        if isinstance(self.residual_loss_fn, AdversarialDecorrelationLoss):
            # Get concepts and residual
            with torch.no_grad():
                (data, concepts), targets = batch
                concept_logits, residual, target_logits = self(data, concepts=concepts)
            if type(concept_logits) == tuple:
                concept_logits = concept_logits[0]
            if type(residual) == tuple:
                residual = residual[0]

            # Calculate discriminator loss and update discriminator
            # Only update discriminator every N batches if specified
            adv_freq = self.hparams.get("adv_decorr_frequency", 1)
            num_steps = self.hparams.get("adv_decorr_num_steps", 1)
            if batch_idx % adv_freq == 0:
                # Train discriminator multiple times per batch for stronger training
                for _ in range(num_steps):
                    disc_loss = self.residual_loss_fn.step(residual, concepts)
                self.log("discriminator_loss", disc_loss, **self.log_kwargs)

            # Calculate decorrelation loss for logging
            decorr_loss = self.residual_loss_fn(residual, concepts)
            self.log("decorrelation_loss", decorr_loss.mean(), **self.log_kwargs)
