from __future__ import annotations

from lib.ccw import EYE
from models.focal_loss import CB_loss
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from typing import Any, Callable, Iterable, Literal

from nn_extensions import Apply, VariableKwargs
from utils import (
    accuracy,
    auroc,
    unwrap,
    zero_loss_fn,
    remove_prefix,
    remove_keys_with_prefix,
)
from nn_extensions import Chain
import numpy as np
import time
import torch_explain as te

### Typing

ConceptBatch = tuple[tuple[Tensor, Tensor], Tensor]  # ((data, concepts), targets)

import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


### Concept Models


class ConceptModel(nn.Module):
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
        cross_attention: nn.Module = None,
        concept_type: Literal["binary", "one_hot", "continuous"] = "binary",
        training_mode: Literal[
            "independent", "sequential", "joint", "intervention_aware"
        ] = "independent",
        additive_residual=False,
        intervention_aware=True,
        freeze_base_model=False,
        residual_dim=None,
        layer_norm_weight=None,
        batch_norm_layer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        concept_network : nn.Module -> (..., concept_dim)
            Concept prediction network
        residual_network : nn.Module -> (..., residual_dim)
            Residual network
        target_network : nn.Module (..., bottleneck_dim) -> (..., num_classes)
            Target network
        base_network : nn.Module
            Optional base network
        bottleneck_layer : nn.Module (..., bottleneck_dim) -> (..., bottleneck_dim)
            Optional bottleneck layer
        concept_type : one of {'binary', 'continuous'}
            Concept type
        training_mode : one of {'independent', 'sequential', 'joint'}
            Training mode (see https://arxiv.org/abs/2007.04612)
        """
        super().__init__()
        cross_attention = cross_attention or Apply(lambda *args: args[1])
        self.freeze_base_model = freeze_base_model
        self.concept_residual_concat = nn.Identity()
        self.layer_norm_weight = layer_norm_weight
        if layer_norm_weight is not None:
            self.layer_norm = nn.LayerNorm(
                residual_dim,
                eps=1e-6,
            )

        def freeze_layers_except_final(model, final_layer_name="fc", freeze_all=False):
            """
            Freezes all layers except the specified final layer.

            Parameters
            ----------
            model : nn.Module
                The network containing layers to freeze.
            final_layer_name : str
                The name of the final layer to keep unfrozen.
            """
            # Iterate through named parameters in the base network
            for name, param in model.named_parameters():
                if freeze_all or (
                    final_layer_name not in name
                ):  # Freeze all layers not containing `final_layer_name`
                    param.requires_grad = False  # Freeze
                else:
                    param.requires_grad = True  # Keep final layer unfrozen
            return model

        if "freeze_backbone" in kwargs and kwargs["freeze_backbone"]:
            base_network = freeze_layers_except_final(base_network)
        if self.freeze_base_model:
            base_network = freeze_layers_except_final(base_network, freeze_all=True)

        self.base_network = VariableKwargs(base_network)

        self.concept_network = VariableKwargs(concept_network)
        self.residual_network = VariableKwargs(residual_network)
        self.target_network = VariableKwargs(target_network)
        self.bottleneck_layer = VariableKwargs(bottleneck_layer)
        self.batch_norm_layer = (
            batch_norm_layer if batch_norm_layer is not None else nn.Identity()
        )
        self.intervention_aware = intervention_aware

        if self.intervention_aware:
            self.concept_rank_model = VariableKwargs(concept_rank_model)
        else:
            self.concept_rank_model = None
        self.cross_attention = VariableKwargs(cross_attention)

        self.concept_type = concept_type
        self.training_mode = training_mode
        self.ccm_mode = "" if "ccm_mode" not in kwargs else kwargs["ccm_mode"]
        if "base_model_ckpt" in kwargs:
            checkpoint_path = kwargs["base_model_ckpt"]
            state_dict = torch.load(checkpoint_path)["state_dict"]
            prefix_to_remove = "concept_model."
            modified_state_dict = remove_prefix(state_dict, prefix_to_remove)
            modified_state_dict = remove_keys_with_prefix(
                modified_state_dict, "concept_loss_fn"
            )
            modified_state_dict = remove_keys_with_prefix(
                modified_state_dict, "residual_loss_fn"
            )
            super().load_state_dict(modified_state_dict, strict=False)
        self.negative_intervention = False
        self.additive_residual = additive_residual
        self.threshold = None

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
        # print(concepts[0])
        # print(x[0])
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
            # x = self.base_network(x)

        if not isinstance(x, torch.Tensor):
            # Handle non-tensor case
            x = x.logits
        x = x.float()
        concept_logits = self.concept_network(x, concepts=concepts)

        if concepts is None:
            concepts = torch.zeros_like(concept_logits).to(x.device)
        

        residual = self.residual_network(x, concepts=concepts)
        if self.layer_norm_weight is not None:
            residual = self.layer_norm_weight * self.layer_norm(residual)
        residual = self.concept_residual_concat(residual)
        # Process concept logits & residual via bottleneck layer
        if not isinstance(unwrap(self.bottleneck_layer), nn.Identity):
            x = torch.cat([concept_logits, residual], dim=-1)
            x = self.bottleneck_layer(x, concepts=concepts)
            concept_dim, residual_dim = concept_logits.shape[-1], residual.shape[-1]
            concept_logits, residual = x.split([concept_dim, residual_dim], dim=-1)

        concept_preds = self.get_concept_predictions(concept_logits)

        # interventions on concepts for intervention_aware training
        if intervention_idxs is None:
            intervention_idxs = torch.tensor(torch.zeros_like(concepts), device=x.device)
        # intervene on concepts
        intervention_idxs = torch.tensor(intervention_idxs, device=intervention_idxs.device)
        concept_preds = (
            concept_preds.detach() * (1 - intervention_idxs)
            + torch.tensor(concepts, device=concept_preds.device) * intervention_idxs
        )

        if self.training and self.training_mode == "independent":
            x_concepts = torch.tensor(concepts, device=concept_preds.device)
        elif self.training and self.training_mode == "sequential":
            x_concepts = concept_preds.detach()
        else:
            # fully joint training
            x_concepts = concept_preds
        attended_residual = self.cross_attention(
            x_concepts, residual, intervention_idxs.detach()
        )
        # print(x_concepts[0])
        # print(residual[0][0])
        if not self.additive_residual:
            x = torch.cat([x_concepts.detach(), attended_residual], dim=-1)
        else:
            assert (
                x_concepts.shape == attended_residual.shape
            ), f"When additive, the concept and residual should have the same shape, but they are {x_concepts.shape} and {attended_residual.shape}"
            x = x_concepts.detach() + attended_residual
        # Get target logits
        # x = self.concept_residual_concat(x)
        # x = self.concept_residual_concat(x)
        #x = self.batch_norm_layer(x)
        target_logits = self.target_network(
            x, concepts=concepts, intervention_idxs=intervention_idxs.detach()
        )
        if target_logits.shape[-1] == 1:
            target_logits = target_logits.squeeze(-1)


        return concept_logits, residual, target_logits

    def get_concept_predictions(self, concept_logits: Tensor) -> Tensor:
        """
        Compute concept predictions from logits.
        """
        if self.concept_type == "binary":
            if self.threshold is not None:
                return (concept_logits.sigmoid() > self.threshold).float()
            return concept_logits.sigmoid()
        elif self.concept_type == "one_hot":
            return torch.nn.functional.gumbel_softmax(
                concept_logits,
                dim=-1,
                hard=True,  # True for discrete one-hot vectors
                tau=1.0,  # Temperature parameter
            )

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
        concept_preds = self.get_concept_predictions(concept_logits)

        concept_preds = (
            concept_preds.detach() * (1 - intervention_idxs)
            + concepts * intervention_idxs
        )
        attended_residual = self.cross_attention(
            concept_preds.detach(), residual, intervention_idxs.detach()
        )
        if not self.additive_residual:
            x = torch.cat([concept_preds.detach(), attended_residual], dim=-1)
        else:
            x = concept_preds.detach() + attended_residual

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
        detach=True,
        get_concept_pred=True,
    ) -> Tensor:
        """
        Compute concept group scores.
        """
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


### Concept Models with PyTorch Lightning


class ConceptLightningModel(pl.LightningModule):
    """
    Lightning module for training a concept model.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        concept_loss_fn: Callable | None = F.binary_cross_entropy_with_logits,
        residual_loss_fn: Callable | None = None,
        target_loss_fn: Callable | None = F.cross_entropy,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        training_intervention_prob=0.25,
        max_horizon=6,
        horizon_rate=1.005,
        initial_horizon=2,
        intervention_discount=1,
        intervention_task_discount=1.1,
        num_rollouts=1,
        include_only_last_trajectory_loss=True,
        intervention_weight=5,
        intervention_task_loss_weight=1,
        weight_decay=0,
        lr_step_size=None,
        lr_gamma=None,
        reg_type=None,
        reg_gamma: float = 1.0,
        lr_scheduler="cosine",
        chosen_optim="adam",
        complete_intervention_weight=0,
        patience=15,
        focal_loss=False,
        torch_explain=False,
        freeze_base_model=False,
        return_auc=False,
        delta=1.0,
        mi_const=1.5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        concept_loss_fn : Callable(concept_logits, concepts) -> loss
            Concept loss function
        residual_loss_fn : Callable(residual, concepts) -> loss
            Residual loss function
        lr : float
            Learning rate
        alpha : float
            Weight for concept loss
        beta : float
            Weight for residual loss
        """
        if "concept_dim" in kwargs and kwargs["concept_dim"] == 0:
            concept_loss_fn = None
        if "residual_dim" in kwargs and kwargs["residual_dim"] == 0:
            residual_loss_fn = None

        super().__init__()
        self.concept_model = concept_model
        self.concept_loss_fn = concept_loss_fn or zero_loss_fn
        self.residual_loss_fn = residual_loss_fn or zero_loss_fn
        self.target_loss_fn = target_loss_fn or zero_loss_fn
        self.focal_loss = focal_loss
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.lamb = 0.0
        self.lamb_lr = delta
        self.MI_const = mi_const
        self.training_intervention_prob = training_intervention_prob
        self.max_horizon = max_horizon
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.include_only_last_trajectory_loss = include_only_last_trajectory_loss
        self.num_rollouts = num_rollouts
        self.intervention_weight = intervention_weight
        self.intervention_task_loss_weight = intervention_task_loss_weight
        self.log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}

        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.reg_type = reg_type
        self.reg_gamma = reg_gamma
        self.lr_scheduler = lr_scheduler
        self.chosen_optim = chosen_optim
        self.num_test_interventions = None
        self.complete_intervention_weight = complete_intervention_weight
        self.patience = patience
        self.intervention_aware = self.concept_model.intervention_aware
        self.torch_explain = torch_explain
        self.return_auc = return_auc

    def dummy_pass(self, loader: Iterable[ConceptBatch]):
        """
        Run a dummy forward pass to handle any uninitialized parameters.

        Parameters
        ----------
        loader : Iterable[ConceptBatch]
            Data loader
        """
        with torch.no_grad():
            (data, concepts), targets = next(iter(loader))
            self.concept_model(data, concepts=concepts)

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass.
        """
        return self.concept_model.forward(*args, **kwargs)

    def _setup_intervention_trajectory(
        self, prev_num_of_interventions, intervention_idxs, free_groups
    ):
        # The limit of how many concepts we can intervene at most
        int_basis_lim = len(intervention_idxs[0])
        # And the limit of how many concepts we will intervene at most during
        # this training round
        horizon_lim = int(self.horizon_limit.detach().cpu().numpy()[0])
        new_intervention_idxs = intervention_idxs.clone()

        if prev_num_of_interventions != int_basis_lim:
            bottom = min(
                horizon_lim,
                int_basis_lim - prev_num_of_interventions - 1,
            )  # -1 so that we at least intervene on one concept
            if bottom > 0:
                initially_selected = np.random.randint(0, bottom)
            else:
                initially_selected = 0

            # Get the maximum size of any current trajectories:
            end_horizon = min(
                int(horizon_lim),
                self.max_horizon,
                int_basis_lim - prev_num_of_interventions - initially_selected,
            )

            # And select the number of steps T we will run the current
            # trajectory for
            current_horizon = np.random.randint(
                1 if end_horizon > 1 else 0,
                end_horizon,
            )

            # At the begining of the trajectory, we start with a total of
            # `initially_selected`` concepts already intervened on. So to
            # indicate that, we will update the intervention_idxs matrix
            # accordingly
            for sample_idx in range(intervention_idxs.shape[0]):
                probs = free_groups[sample_idx, :].detach().cpu().numpy()
                probs = probs / np.sum(probs)
                new_intervention_idxs[
                    sample_idx,
                    np.random.choice(
                        int_basis_lim,
                        size=initially_selected,
                        replace=False,
                        p=probs,
                    ),
                ] = 1

            discount = 1
            trajectory_weight = 0
            for i in range(current_horizon):
                trajectory_weight += discount
                discount *= self.intervention_discount
            task_discount = 1
            task_trajectory_weight = 1
            for i in range(current_horizon):
                task_discount *= self.intervention_task_discount
                if (not self.include_only_last_trajectory_loss) or (
                    i == current_horizon - 1
                ):
                    task_trajectory_weight += task_discount
            task_discount = 1
        else:
            # Else we will peform no intervention in this training step!
            current_horizon = 0
            task_discount = 1
            task_trajectory_weight = 1
            trajectory_weight = 1

        return (
            current_horizon,
            task_discount,
            task_trajectory_weight,
            trajectory_weight,
            new_intervention_idxs,
        )

    def get_concept_mask(
        self,
        concept_logits,
        concepts,
        residual,
        prev_intervention_idxs,
        concept_group_scores,
        targets,
    ):
        target_int_losses = torch.ones(
            concept_group_scores.shape,
        ).to(
            concepts.device
        ) * (-np.Inf)
        for target_concept in range(target_int_losses.shape[-1]):
            new_int = torch.zeros(prev_intervention_idxs.shape).to(
                prev_intervention_idxs.device
            )
            new_int[:, target_concept] = 1
            updated_int = torch.clamp(
                prev_intervention_idxs.detach() + new_int,
                0,
                1,
            )

            target_logits = self.concept_model.calc_target_preds(
                concept_logits=concept_logits,
                residual=residual,
                concepts=concepts,
                intervention_idxs=updated_int,
                get_concept_pred=False
            )
            current_loss = F.cross_entropy(target_logits, targets, reduction="none")
            target_int_losses[:, target_concept] = current_loss
        # find the lable with the lowest loss (skyline oracle for the intervention)
        # this becomes the target for behavioral cloning
        target_int_labels = torch.argmin(target_int_losses, -1)
        # find the label the model predicts (greedy policy prediction)
        pred_int_labels = concept_group_scores.argmax(-1)
        curr_acc = (pred_int_labels == target_int_labels).float().mean()
        return target_int_labels, curr_acc

    def rollout_loss_fn(
        self,
        batch: ConceptBatch,
        outputs: tuple[Tensor, Tensor, Tensor],
        prev_intervention_idxs: Tensor,
    ) -> Tensor:
        (data, concepts), targets = batch
        concept_logits, residual, target_logits = outputs

        free_groups = 1 - prev_intervention_idxs
        prev_num_of_interventions = int(
            np.max(
                np.sum(
                    prev_intervention_idxs.detach().cpu().numpy(),
                    axis=-1,
                ),
                axis=-1,
            )
        )
        (
            current_horizon,
            task_discount,
            task_trajectory_weight,
            trajectory_weight,
            prev_intervention_idxs,
        ) = self._setup_intervention_trajectory(
            prev_num_of_interventions,
            prev_intervention_idxs,
            free_groups,
        )
        discount = 1

        intervention_task_loss = F.cross_entropy(target_logits, targets)
        intervention_task_loss = intervention_task_loss / task_trajectory_weight
        if type(concept_logits) == tuple:
            concept_logits_device = concept_logits[0].device
        else:
            concept_logits_device = concept_logits.device
        intervention_loss = torch.tensor(0.0, device=concept_logits_device)

        int_mask_accuracy = 0.0 if current_horizon else -1

        for _ in range(self.num_rollouts):
            # And as many steps in the trajectory as suggested
            for i in range(current_horizon):
                # And generate a probability distribution over previously
                # unseen concepts to indicate which one we should intervene
                # on next!
                concept_group_scores = self.concept_model.calc_concept_group(
                    concept_logits=concept_logits,
                    residual=residual,
                    concepts=concepts,
                    intervention_idxs=prev_intervention_idxs,
                    train=True,
                )

                target_int_labels, curr_acc = self.get_concept_mask(
                    concept_logits=concept_logits,
                    residual=residual,
                    concepts=concepts,
                    prev_intervention_idxs=prev_intervention_idxs,
                    concept_group_scores=concept_group_scores,
                    targets=targets,
                )
                int_mask_accuracy += curr_acc / current_horizon
                new_loss = torch.nn.CrossEntropyLoss()(
                    concept_group_scores, target_int_labels.detach()
                )
                intervention_loss += discount * new_loss / trajectory_weight

                discount *= self.intervention_discount
                task_discount *= self.intervention_task_discount
                if self.intervention_weight == 0:
                    selected_groups = torch.FloatTensor(
                        np.eye(concept_group_scores.shape[-1])[
                            np.random.choice(
                                concept_group_scores.shape[-1],
                                size=concept_group_scores.shape[0],
                            )
                        ]
                    ).to(concept_group_scores.device)
                else:
                    selected_groups = torch.nn.functional.gumbel_softmax(
                        concept_group_scores,
                        dim=-1,
                        hard=True,
                        tau=1,
                    )
                prev_intervention_idxs += selected_groups

                if (not self.include_only_last_trajectory_loss) or (
                    i == (current_horizon - 1)
                ):
                    rollout_y_logits = self.concept_model.calc_target_preds(
                        concept_logits=concept_logits,
                        residual=residual,
                        concepts=concepts,
                        intervention_idxs=prev_intervention_idxs,
                        get_concept_pred=False
                    )
                    rollout_y_loss = F.cross_entropy(rollout_y_logits, targets)

                    intervention_task_loss += (
                        task_discount * rollout_y_loss / task_trajectory_weight
                    )

            if self.horizon_limit.detach().cpu().numpy()[0] < (
                (concepts.shape[-1] + 1)
            ):
                self.horizon_limit *= self.horizon_rate

        intervention_loss = intervention_loss / self.num_rollouts
        if intervention_loss.requires_grad:
            self.log("intervention_loss", intervention_loss, **self.log_kwargs)

        intervention_task_loss = intervention_task_loss / self.num_rollouts
        if intervention_task_loss.requires_grad:
            self.log("task_loss", intervention_task_loss, **self.log_kwargs)

        int_mask_accuracy = int_mask_accuracy / self.num_rollouts
        self.log("int_mask_accuracy", int_mask_accuracy, **self.log_kwargs)

        return intervention_loss, intervention_task_loss, int_mask_accuracy

    def concept_residual_loss_fn(
        self,
        batch: ConceptBatch,
        outputs: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Compute loss.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        outputs : tuple[Tensor, Tensor, Tensor]
            Concept model outputs (concept_logits, residual, target_logits)
        """
        (data, concepts), targets = batch
        concept_logits, residual, target_logits = outputs
        if type(concept_logits) == tuple:
            concept_logits, concept_logvars = concept_logits
        else:
            concept_logvars = None
        if type(residual) == tuple and len(residual) == 2:
            # full probabilistc
            residual, residual_logvars = residual
        elif type(residual) == tuple and len(residual) == 4:
            # partial probabilistic
            residual, means, stds, logprob = residual
        else:
            residual_logvars = None

        # Concept loss
        if self.focal_loss:
            no_of_classes = concept_logits.shape[-1]
            # beta = 0.9999
            # gamma = 2.0
            # concept_loss = CB_loss(
            #     concepts,
            #     concept_logits,
            #     samples_per_cls,
            #     no_of_classes,
            #     "focal",
            #     beta,
            #     gamma,
            # )
        else:
            assert (
                concept_logits.shape == concepts.shape
            ), f"concept_logits.shape, concepts.shape: {concept_logits.shape, concepts.shape}"
            concept_loss = self.concept_loss_fn(concept_logits, concepts)
            # if concept_logvars is None:
            #     concept_loss = self.concept_loss_fn(concept_logits, concepts)
            # else:
            #     pos_weight =  self.concept_loss_fn.pos_weight
            #     bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
            #     per_sample_bce = bce_loss(concept_logits, concepts)

            #     # If per_sample_bce is [batch_size, num_classes], take mean over classes
            #     if len(per_sample_bce.shape) > 1:
            #         per_sample_bce = per_sample_bce.mean(dim=1)

            #     # Calculate certainty weights from concept logvars
            #     # Lower logvar = higher certainty = higher weight
            #     # We take the mean across all concepts for each sample
            #     certainty_weights = torch.exp(-0.5 * concept_logvars.mean(dim=1))

            #     # Normalize weights to sum to batch size (to keep overall loss magnitude similar)
            #     batch_size = certainty_weights.size(0)
            #     normalized_weights = certainty_weights * (batch_size / certainty_weights.sum())

            #     # Apply weights to the BCE loss
            #     concept_loss = (per_sample_bce * normalized_weights).mean()
            # try:
            #     concept_loss = self.concept_loss_fn(concept_logits, concepts)
            # except:
            #     breakpoint()
        if concept_loss.requires_grad:
            self.log("concept_loss", concept_loss, **self.log_kwargs)

        # Residual loss
        residual_loss = self.residual_loss_fn(residual, concepts)
        if residual_loss.requires_grad:
            self.log("residual_loss", residual_loss, **self.log_kwargs)

        # Regularization loss
        if self.reg_type == "l1":
            if type(self.concept_model.target_network) == Chain:
                # If the target network is a Chain, the target network is the first module in the chain
                net_y = self.concept_model.target_network[1].module
            else:
                net_y = self.concept_model.target_network.module
            if not isinstance(net_y, nn.Linear):
                net_y = net_y[1]
            A = net_y.weight
            concept_dim = concept_logits.shape[-1]
            A_residual = A[
                :, concept_dim:
            ]  # Only keep the weights corresponding to the residual transform

            def compute_l1_loss(w):
                return torch.abs(w).sum() / w.shape[0]

            reg_loss = compute_l1_loss(A_residual)

        elif self.reg_type == "eye":
            if type(self.concept_model.target_network) == Chain:
                # If the target network is a Chain, the target network is the first module in the chain
                net_y = self.concept_model.target_network[1].module
            else:
                net_y = self.concept_model.target_network.module
            if not isinstance(net_y, nn.Linear):
                net_y = net_y[1]
            device = residual.device
            r = torch.cat(
                [torch.ones(concept_logits.shape[1]), torch.zeros(residual.shape[1])]
            ).to(device)
            reg_loss = EYE(r, net_y.weight.abs().mean(0))
        else:
            reg_loss = torch.tensor(0.0)
        if reg_loss.requires_grad:
            self.log("reg_loss", reg_loss, **self.log_kwargs)

        # # Target loss
        # target_loss = F.cross_entropy(target_logits, targets)
        # if target_loss.requires_grad:
        #     self.log("target_loss", target_loss, **self.log_kwargs)
        return concept_loss, residual_loss, reg_loss

        # return (
        #     + (self.alpha * concept_loss)
        #     + (self.beta * residual_loss)
        #     + (self.reg_gamma * reg_loss)
        # )

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        """

        if self.chosen_optim == "adam":
            optimizer = torch.optim.Adam(
                self.concept_model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.chosen_optim == "adamw":
            optimizer = torch.optim.AdamW(
                self.concept_model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.chosen_optim == "sgd":
            optimizer = torch.optim.SGD(
                self.concept_model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.chosen_optim}")

        if self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_step_size,
                gamma=self.lr_gamma,
            )
        elif self.lr_scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100,
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def step(
        self,
        batch: ConceptBatch,
        split: Literal["train", "val", "test"],
        intervention_idxs: Tensor | None = None,
    ) -> Tensor:
        """
        Training / validation / test step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        split : one of {'train', 'val', 'test'}
            Dataset split
        """
        (data, concepts), targets = batch
        t1 = time.time()
        if self.training and self.intervention_aware:
            mask = torch.bernoulli(
                torch.ones_like(concepts[0]) * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (concepts.shape[0], 1),
            )
        elif intervention_idxs is not None:
            intervention_idxs = intervention_idxs
        else:
            intervention_idxs = torch.zeros_like(concepts)

        outputs = self.concept_model(
            data,
            concepts=concepts,
            intervention_idxs=torch.ones_like(intervention_idxs),
        )
        concept_logits, residual, intervention_target_logits = outputs

        if type(concept_logits) == tuple:
            concept_logits, concept_logvars = concept_logits

        # Rollout loss
        if self.intervention_aware and (
            self.intervention_weight > 0 or self.intervention_task_loss_weight > 0
        ):
            outputs = self.concept_model(
                data, concepts=concepts, intervention_idxs=intervention_idxs
            )
            _, residual, target_logits = outputs
            intervention_loss, intervention_task_loss, int_mask_accuracy = (
                self.rollout_loss_fn(batch, outputs, intervention_idxs)
            )
        else:
            intervention_loss = torch.tensor(0.0, device=concept_logits.device)
            intervention_task_loss = torch.tensor(0.0, device=concept_logits.device)
            nt_mask_accuracy = -1

        # Concept / Residual Loss
        concept_loss, residual_loss, reg_loss = self.concept_residual_loss_fn(
            batch, outputs
        )

        if hasattr(self.concept_model, "x_r_mi_loss"):
            r_c_mi_loss = self.concept_model.x_r_mi_loss(
                residual,
            )
            constraint = self.MI_const - r_c_mi_loss
            self.lamb += self.lamb_lr * constraint.detach().cpu().numpy().item()

            self.log(
                f"{split}_r_c_mi_loss",
                r_c_mi_loss,
                **self.log_kwargs,
            )
        else:
            r_c_mi_loss = torch.tensor(0.0, device=concept_logits.device)
            constraint = torch.tensor(0.0, device=concept_logits.device)

        complete_intervention_loss = self.target_loss_fn(
            intervention_target_logits, targets
        )

        loss = (
            self.alpha * concept_loss
            + self.beta * residual_loss
            + self.lamb * constraint
            + self.reg_gamma * reg_loss
            + self.intervention_weight * intervention_loss
            + self.intervention_task_loss_weight * intervention_task_loss
            + self.complete_intervention_weight * complete_intervention_loss
        )

        if self.torch_explain:
            entropy_loss = te.nn.functional.entropy_logic_loss(
                self.concept_model.target_network.module
            )
            loss += 0.0001 * entropy_loss
            self.log("entropy_loss", entropy_loss, **self.log_kwargs)

        self.log(f"{split}_loss", loss, **self.log_kwargs)

        def make_sure_torch(x):
            if type(x) == torch.Tensor:
                return x
            else:
                return torch.tensor(x, dtype=x.dtype)

        if split != "train":
            # Track baseline accuracy
            outputs = self.concept_model(
                data, concepts=concepts, intervention_idxs=intervention_idxs
            )
            _, _, target_logits_baseline = outputs
            #target_logits_baseline_torch = make_sure_torch(target_logits_baseline)
            acc_base = accuracy(torch.tensor(target_logits_baseline), targets)
            self.log(f"{split}_acc", acc_base, **self.log_kwargs)
            if self.return_auc:
                auroc_score_baseline = auroc(target_logits_baseline, targets)
                self.log(f"{split}_auroc", auroc_score_baseline, **self.log_kwargs)

                auroc_score = auroc(intervention_target_logits, targets)
                self.log(f"{split}_intervention_auroc", auroc_score, **self.log_kwargs)
        #intervention_target_logits_torch = make_sure_torch(intervention_target_logits)
        #intervention_acc = accuracy(torch.tensor(intervention_target_logits), targets)
        #print("H")
        #print(intervention_acc)
        #print(torch.mean((torch.argmax(intervention_target_logits, dim=-1) == targets).to(torch.float32)))
        intervention_acc = torch.mean((torch.argmax(intervention_target_logits, dim=-1) == targets).to(torch.float32))
        #print(intervention_target_logits.shape)
        print(torch.argmax(intervention_target_logits, dim=-1)[0])
        print(targets[0])
        print(intervention_target_logits.shape)
        self.log(f"{split}_intervention_acc", intervention_acc, **self.log_kwargs)

        # Track concept accuracy
        if (
            self.concept_model.concept_type == "binary"
            or self.concept_model.concept_type == "one_hot"
        ):
            concept_logits_torch = make_sure_torch(concept_logits)
            concept_acc = accuracy(concept_logits_torch, concepts, task="binary")
            self.log(f"{split}_concept_acc", concept_acc, **self.log_kwargs)
        else:
            concept_rmse = F.mse_loss(concept_logits, concepts).sqrt()
            self.log(f"{split}_concept_rmse", concept_rmse, **self.log_kwargs)
        t2 = time.time()
        print(t2 - t1)

        return loss

    def training_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        return self.step(batch, split="train")

    def validation_step(self, batch: ConceptBatch, batch_idx: int) -> Tensor:
        """
        ValConceptModelidation step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        with torch.no_grad():
            return self.step(batch, split="val")

    def perform_k_interventions(
        self,
        batch: ConceptBatch,
        split: Literal["train", "val", "test"],
        k: int,
    ) -> Tensor:
        """
        Training / validation / test step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        split : one of {'train', 'val', 'test'}
            Dataset split
        """
        (data, concepts), targets = batch

        intervention_idxs = torch.zeros_like(concepts)

        if self.intervention_aware:
            # start with no interventions

            outputs = self.concept_model(
                data, concepts=concepts, intervention_idxs=intervention_idxs
            )
            concept_logits, residual, target_logits = outputs

        for i in range(k):
            if not self.intervention_aware:
                # Randomly select an intervention index
                available_mask = (intervention_idxs == 0).float()

                # Set very low probability for already intervened concepts
                scores = torch.where(
                    available_mask == 1,
                    torch.zeros_like(intervention_idxs),
                    torch.ones_like(intervention_idxs) * (-1000),
                )

                # Use gumbel_softmax for random selection, just like in intervention_aware case
                selected_groups = torch.nn.functional.gumbel_softmax(
                    scores,
                    dim=-1,
                    hard=True,
                    tau=1,
                )
            else:
                # And generate a probability distribution over previously
                # unseen concepts to indicate which one we should intervene
                # on next!
                concept_group_scores = self.concept_model.calc_concept_group(
                    concept_logits=concept_logits,
                    residual=residual,
                    concepts=concepts,
                    intervention_idxs=intervention_idxs,
                    train=True,
                )
                concept_group_scores = torch.where(
                    intervention_idxs == 1,
                    torch.ones(intervention_idxs.shape).to(intervention_idxs.device)
                    * (-1000),
                    concept_group_scores,
                )
                selected_groups = torch.nn.functional.gumbel_softmax(
                    concept_group_scores,
                    dim=-1,
                    hard=True,
                    tau=1,
                )

            intervention_idxs += selected_groups
            intervention_idxs = torch.clamp(
                intervention_idxs,
                0,
                1,
            )

        return intervention_idxs

    def test_step(
        self, batch: ConceptBatch, batch_idx: int, return_intervention_idxs=False
    ) -> Tensor:
        """
        Test step.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        t0 = time.time()
        if self.num_test_interventions is not None:
            intervention_idxs = self.perform_k_interventions(
                batch, split="test", k=self.num_test_interventions
            )
        else:
            intervention_idxs = None
        print(intervention_idxs)

        # print(f"Forward time {t2 - t1}")
        with torch.no_grad():
            res = self.step(batch, split="test", intervention_idxs=intervention_idxs)
            if return_intervention_idxs:
                return (
                    res,
                    intervention_idxs,
                )
            else:
                return res

    def forward_intervention(
        self, batch: ConceptBatch, batch_idx: int, return_intervention_idxs=False
    ) -> Tensor:
        if self.num_test_interventions is not None:
            intervention_idxs = self.perform_k_interventions(
                batch, split="test", k=self.num_test_interventions
            )
        else:
            intervention_idxs = None

        (data, concepts), targets = batch
        forw = self.concept_model.forward(
            data, concepts, intervention_idxs=intervention_idxs
        )
        return (
            forw,
            intervention_idxs,
        )
