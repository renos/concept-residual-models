from __future__ import annotations

from lib.ccw import EYE
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from typing import Any, Callable, Iterable, Literal

from nn_extensions import VariableKwargs
from utils import accuracy, unwrap, zero_loss_fn, remove_prefix, remove_keys_with_prefix
from nn_extensions import Chain
import numpy as np

### Typing

ConceptBatch = tuple[tuple[Tensor, Tensor], Tensor]  # ((data, concepts), targets)
from .base import ConceptModel, ConceptLightningModel


class ConceptEmbeddingModel(ConceptModel):

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
            concept_network=None,
            residual_network=None,
            target_network=target_network,
            base_network=base_network,
            bottleneck_layer=bottleneck_layer,
            concept_rank_model=concept_rank_model,
            concept_type=concept_type,
            training_mode=training_mode,
            **kwargs,
        )
        self.concept_prob_generators = concept_network
        self.concept_context_generators = residual_network
        self.negative_intervention = False

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
        # Get concept logits & residual
        if self.negative_intervention:
            concepts = 1 - concepts

        if x.device.type == "cpu":
            model_float = self.base_network.to(torch.float32)
            x = model_float(x)

        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = self.base_network(x)
            # x = self.base_network(x)
        x = x.float()  # Ensure x is float for precision
        contexts = []
        c_sem = []
        concept_logits = []

        # First predict all the concept probabilities
        for i, context_gen in enumerate(self.concept_context_generators):
            prob_gen = self.concept_prob_generators[i]
            context = context_gen(x)
            prob = prob_gen(context)
            contexts.append(torch.unsqueeze(context, dim=1))
            c_sem.append(torch.nn.Sigmoid()(prob))
            concept_logits.append(prob)
        c_sem = torch.cat(c_sem, axis=-1)
        concept_logits = torch.cat(concept_logits, axis=-1)
        contexts = torch.cat(contexts, axis=1)

        r_dim = contexts.shape[-1]
        pos_embedding = contexts[:, :, : r_dim // 2]
        neg_embedding = contexts[:, :, r_dim // 2 :]

        # interventions on concepts for intervention_aware training
        if concepts is None:
            concepts = torch.zeros_like(concept_logits).to(x.device)

        if intervention_idxs is None:
            intervention_idxs = torch.zeros_like(concepts)
        # intervene on concepts
        concept_preds = (
            c_sem.detach() * (1 - intervention_idxs) + concepts * intervention_idxs
        )

        if self.training and self.training_mode == "independent":
            x_concepts = concepts
        elif self.training and self.training_mode == "sequential":
            x_concepts = concept_preds.detach()
        else:
            # fully joint training
            x_concepts = concept_preds
        x = pos_embedding * torch.unsqueeze(
            x_concepts, dim=-1
        ) + neg_embedding * torch.unsqueeze(1 - x_concepts, dim=-1)
        batch_size, n_concepts, emb_size = x.shape
        x = x.view((batch_size, emb_size * n_concepts))

        x = self.concept_residual_concat(x)

        # Get target logits
        target_logits = self.target_network(x, concepts=concepts)

        return concept_logits, contexts, target_logits

    def get_concept_predictions(self, concept_logits: Tensor) -> Tensor:
        """
        Compute concept predictions from logits.
        """
        if self.concept_type == "binary":
            return concept_logits.sigmoid()

        return concept_logits

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

        concept_preds = (
            concept_logits.detach().sigmoid() * (1 - intervention_idxs)
            + concepts * intervention_idxs
        )
        r_dim = residual.shape[-1]
        pos_embedding = residual[:, :, : r_dim // 2]
        neg_embedding = residual[:, :, r_dim // 2 :]
        x = pos_embedding * torch.unsqueeze(
            concept_preds, dim=-1
        ) + neg_embedding * torch.unsqueeze(1 - concept_preds, dim=-1)
        x = x.view((x.shape[0], -1))

        rank_input = torch.concat(
            [x, intervention_idxs],
            dim=-1,
        ).detach()

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
    ) -> Tensor:
        """
        Compute concept group scores.
        """

        concept_preds = (
            concept_logits.detach().sigmoid() * (1 - intervention_idxs)
            + concepts * intervention_idxs
        )
        r_dim = residual.shape[-1]
        pos_embedding = residual[:, :, : r_dim // 2]
        neg_embedding = residual[:, :, r_dim // 2 :]
        x = pos_embedding * torch.unsqueeze(
            concept_preds, dim=-1
        ) + neg_embedding * torch.unsqueeze(1 - concept_preds, dim=-1)
        x = x.view((x.shape[0], -1))

        target_logits = self.target_network(x)
        return target_logits
