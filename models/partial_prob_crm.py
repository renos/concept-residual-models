from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal, Tuple
from .base import ConceptModel, ConceptLightningModel
from utils import (
    accuracy,
    auroc,
    unwrap,
    zero_loss_fn,
    remove_prefix,
    remove_keys_with_prefix,
)
import numpy as np


class PartialProbabilisticConceptModel(ConceptModel):
    """
    Base class for concept models.

    Attributes
    ----------
    base_network : nn.Module
        Network that pre-processes input
    concept_network : nn.Module
        Network that takes the output of `base_network` and
        generates concept logits
    residual_network : nn.Module
        Network that takes the output of `base_network` and
        generates a residual vector
    bottleneck_layer : nn.Module
        Network that post-processes the concatenated output of
        `concept_network` and `residual_network`
    target_network : nn.Module
        Network that takes the output of `bottleneck_layer` and
        generates target logits
    training_mode : one of {'independent', 'sequential', 'joint'}
        Training mode (see https://arxiv.org/abs/2007.04612)
    """

    def __init__(
        self,
        concept_network: nn.Module,
        residual_network: nn.Module,
        target_network: nn.Module,
        base_network: nn.Module = nn.Identity(),
        bottleneck_layer: nn.Module = nn.Identity(),
        concept_rank_model: nn.Module = nn.Identity(),
        concept_type: Literal["binary", "continuous"] = "binary",
        training_mode: Literal[
            "independent", "sequential", "joint", "intervention_aware"
        ] = "independent",
        **kwargs,
    ):
        super().__init__(
            concept_network=concept_network,
            residual_network=residual_network,
            target_network=target_network,
            base_network=base_network,
            bottleneck_layer=bottleneck_layer,
            concept_rank_model=concept_rank_model,
            concept_type=concept_type,
            training_mode=training_mode,
            **kwargs,
        )

    def forward(
        self,
        x: Tensor,
        concepts: Tensor | None = None,
        intervention_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor
        concepts : Tensor or None
            Ground truth concept values
        intervention_idxs : Tensor or None
            Indices of interventions

        Returns
        -------
        concept_logits : Tensor
            Concept logits
        residual : Tensor
            Residual vector
        target_logits : Tensor
            Target logits
        """
        # if negative intervention, invert accurate concepts
        if concepts is not None and self.negative_intervention:
            concepts = 1 - concepts

        # Get concept logits & residual
        if x.device.type == "cpu":
            model_float = self.base_network.to(torch.float32)
            x = model_float(x)

        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = self.base_network(x)

        if not isinstance(x, torch.Tensor):
            # Handle non-tensor case
            x = x.logits
        x = x.float()

        concept_logits = self.concept_network(x, concepts=concepts)

        if concepts is None:
            concepts = torch.zeros_like(concept_logits).to(x.device)

        output = self.residual_network(x, concepts=concepts)
        residual_logits, residual_logvars_before_softplus = torch.split(
            output, output.shape[-1] // 2, dim=-1
        )
        residual_logvars = torch.nn.functional.softplus(
            residual_logvars_before_softplus
        ).clamp(min=1e-6)

        if self.training:
            # Sample during training
            eps = torch.randn_like(residual_logvars)
            z = residual_logits + eps * residual_logvars

            # Calculate log probability if needed
            distr = torch.distributions.normal.Normal(residual_logits, residual_logvars)
            z_prob = z
            # Optional: detach z for log probability calculation
            # if self.DETACH:
            #     z_prob = z.detach()
            logprob = distr.log_prob(z_prob).sum(dim=1)  # batch_sz

            residual = (z, residual_logits, residual_logvars, logprob)
        else:
            # Use means directly during inference
            z = residual_logits
            distr = torch.distributions.normal.Normal(residual_logits, residual_logvars)
            z_prob = z
            # Optional: detach z for log probability calculation
            # if self.DETACH:
            #     z_prob = z.detach()
            logprob = distr.log_prob(z_prob).sum(dim=1)  # batch_sz
            residual = (z, residual_logits, residual_logvars, logprob)

        # Process concept logits & residual via bottleneck layer
        if not isinstance(unwrap(self.bottleneck_layer), nn.Identity):
            raise NotImplementedError("Non-identity bottleneck not supported for PartialProbabilisticConceptModel")

        # interventions on concepts for intervention_aware training
        if intervention_idxs is None:
            intervention_idxs = torch.zeros_like(concepts)
        target_logits = self.calc_target_preds(
            concept_logits,
            residual,
            concepts,
            intervention_idxs,
        )
        return (
            concept_logits,
            residual,
            target_logits,
        )

    def calc_concept_group(
        self,
        concept_logits: Tensor,
        residual: Tensor,
        concepts: Tensor,
        intervention_idxs: Tensor,
        train: bool = False,
    ) -> Tensor:
        """
        Compute concept group scores.
        """
        if type(residual) == tuple:
            residual, _, _, _ = residual

        concept_preds = self.get_concept_predictions(concept_logits)

        concept_preds = (
            concept_preds.detach() * (1 - intervention_idxs)
            + concepts * intervention_idxs
        )

        attended_residual = self.cross_attention(
            concept_preds.detach(), residual, intervention_idxs.detach()
        )
        if not self.additive_residual:
            x = torch.cat(
                [
                    concept_preds.detach(),
                    attended_residual,
                ],
                dim=-1,
            )
        else:
            x = concept_preds.detach() + attended_residual

        # rank_input = torch.concat(
        #     [x, intervention_idxs],
        #     dim=-1,
        # ).detach()
        rank_input = x.detach()

        next_concept_group_scores = self.concept_rank_model(rank_input)
        if train:
            return next_concept_group_scores

        # zero out the scores of the concepts that have already been intervened on
        next_concept_group_scores = torch.where(
            intervention_idxs == 1,
            torch.ones(intervention_idxs.shape).to(intervention_idxs.device) * (-1000),
            next_concept_group_scores,
        )
        # return the softmax of the scores
        return torch.nn.functional.softmax(
            next_concept_group_scores,
            dim=-1,
        )

    def calc_target_preds(
        self,
        concept_logits: Tensor,
        residual: Tensor,
        concepts: Tensor,
        intervention_idxs: Tensor,
        train: bool = False,
        detach=True,
        get_concept_pred=True,
    ) -> Tensor:
        """
        Compute concept group scores.
        """
        if type(residual) == tuple:
            residual, _, _, _ = residual
        if get_concept_pred:
            concept_preds = self.get_concept_predictions(concept_logits)
        else:
            concept_preds = concept_logits
        if detach:
            concept_preds = (
                concept_preds.detach() * (1 - intervention_idxs)
                + concepts * intervention_idxs
            )
        else:
            concept_preds = (
                concept_preds * (1 - intervention_idxs) + concepts * intervention_idxs
            )
        attended_residual = self.cross_attention(
            concept_preds, residual, intervention_idxs.detach()
        )
        if not self.additive_residual:
            x = torch.cat([concept_preds, attended_residual], dim=-1)
        else:
            x = concept_preds + attended_residual

        target_logits = self.target_network(
            x, concepts=concepts, intervention_idxs=intervention_idxs.detach()
        )
        if target_logits.shape[-1] == 1:
            target_logits = target_logits.squeeze(-1)
        return target_logits

    def log_marg_prob(self, z, means, stds, jensen, DETACH=False):
        batch_sz, L = z.shape
        batch_sz2 = batch_sz

        # for each target, pass through each mean
        means = means.unsqueeze(0).expand(batch_sz, batch_sz2, L)
        stds = stds.unsqueeze(0).expand(batch_sz, batch_sz2, L)

        z = z.unsqueeze(1).expand(batch_sz, batch_sz2, L)

        distr = torch.distributions.normal.Normal(means, stds)
        z_prob = z
        if DETACH:
            z_prob = z.detach()
        logprob = distr.log_prob(z_prob)
        assert logprob.shape == (batch_sz, batch_sz2, L)

        logprob = logprob.sum(
            dim=2
        )  # batch_sz, batch_sz2, logprob of each code, was missing before!
        if jensen:
            log_margprob = logprob.mean(dim=1)  # est
        else:
            log_margprob = -np.log(batch_sz2) + torch.logsumexp(logprob, dim=1)

        assert log_margprob.shape == (batch_sz,)

        return log_margprob  # batch_sz

    def x_r_mi_loss(
        self,
        residual: Tensor,
        train: bool = False,
    ):
        if type(residual) == tuple:
            residual, means, stds, logprob = residual

        log_marg_prob = self.log_marg_prob(residual, means, stds, jensen=False)

        return (logprob - log_marg_prob).mean()
