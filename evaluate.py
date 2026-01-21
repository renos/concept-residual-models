"""
Evaluation script for concept residual models.

This module provides comprehensive evaluation functions for analyzing trained
concept bottleneck models including:
- Accuracy testing with concept interventions (positive and negative)
- Concept/residual randomization tests
- Cross-correlation analysis between concepts and residuals
- Mutual information estimation
- Concept prediction from residuals
- DeepLift Shapley value attribution analysis
- TCAV (Testing with Concept Activation Vectors) analysis
- Counterfactual evaluation

Usage:
    python evaluate.py --exp-dir ./saved --mode accuracy pos_intervention
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import ray.train
import torch
import torch.nn as nn

from tqdm import tqdm

# Ensure SLURM environment variables don't interfere with Ray
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]
if "SLURM_JOB_NAME" in os.environ:
    del os.environ["SLURM_JOB_NAME"]

from copy import deepcopy
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray import air, tune
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import DATASET_INFO, get_concept_loss_fn
from lightning_ray import LightningTuner
from nn_extensions import Chain
from models import ConceptLightningModel
from models.mutual_info import MutualInformationLoss
from models.posthoc_concept_pred import (
    ConceptResidualConceptPred,
    ConceptEmbeddingConceptPred,
)
from train import make_concept_model, make_datamodule
from utils import cross_correlation, set_cuda_visible_devices


### Interventions


class Randomize(nn.Module):
    """
    Shuffle data along the batch dimension.
    """

    def forward(self, x: Tensor, *args, **kwargs):
        return x[torch.randperm(len(x))]


class Intervention(nn.Module):
    """
    Intervene on a random subset of concepts.
    """

    def __init__(self, num_interventions: int, negative: bool = False):
        """
        Parameters
        ----------
        num_interventions : int
            Number of concepts to intervene on
        negative : bool
            Whether to intervene with incorrect concept values
        """
        super().__init__()
        self.num_interventions = num_interventions
        self.negative = negative

    def forward(self, x: Tensor, concepts: Tensor):
        if self.negative:
            concepts = 1 - concepts  # flip binary concepts to opposite values

        concept_dim = concepts.shape[-1]
        idx = torch.randperm(concept_dim)[: self.num_interventions]
        x[..., idx] = concepts[..., idx]
        return x


### Evaluations


def test(model: pl.LightningModule, loader: DataLoader):
    """
    Test model.

    Parameters
    ----------
    model : pl.LightningModule
        Model to test
    loader : DataLoader
        Test data loader
    """
    trainer = pl.Trainer(
        accelerator="cuda" if MPSAccelerator.is_available() else "auto",
        enable_progress_bar=True,
    )
    return trainer.test(model, loader)[0]


def test_interventions(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    concept_dim: int,
    dataset: str,
    negative: bool,
    max_samples: int = 4,
) -> float:
    """
    Test model accuracy with concept interventions.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Dimension of concept vector
    negative : bool
        Whether to intervene with incorrect concept values
    max_samples : int
        Maximum number of interventions to test (varying the # of concepts intervened on)
    """
    x = np.linspace(
        0, concept_dim + 1, num=min(concept_dim + 2, max_samples), dtype=int
    )
    # x = x[::-1]
    y = np.zeros(len(x))
    for i, num_interventions in enumerate(x):
        # intervention = Intervention(num_interventions, negative=negative)
        new_model = deepcopy(model)
        new_model.num_test_interventions = num_interventions
        new_model.concept_model.negative_intervention = negative

        # new_model.concept_model.target_network = Chain(
        #     intervention,
        #     new_model.concept_model.target_network,
        # )
        results = test(new_model, test_loader)
        if dataset != "mimic_cxr":
            y[i] = results["test_acc"]
        else:
            y[i] = results["test_auroc"]

    return {"x": x, "y": y}


def test_threshold_fitting(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    dataset: str,
) -> float:
    """
    Test model accuracy with concept interventions.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    concept_dim : int
        Dimension of concept vector
    negative : bool
        Whether to intervene with incorrect concept values
    max_samples : int
        Maximum number of interventions to test (varying the # of concepts intervened on)
    """
    x = np.linspace(0.45, 0.55, num=10, dtype=float)
    # x = x[::-1]
    y = np.zeros(len(x))
    for i, threshold in enumerate(x):
        # intervention = Intervention(num_interventions, negative=negative)
        new_model = deepcopy(model)
        new_model.concept_model.threshold = threshold

        # new_model.concept_model.target_network = Chain(
        #     intervention,
        #     new_model.concept_model.target_network,
        # )
        results = test(new_model, test_loader)
        if dataset != "mimic_cxr":
            y[i] = results["test_acc"]
        else:
            y[i] = results["test_auroc"]

    return {"x": x, "y": y}


def test_random_concepts(
    model: ConceptLightningModel, test_loader: DataLoader, dataset: str
) -> float:
    """
    Test model accuracy with randomized concept predictions.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    # Shuffle data
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=True,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory,
    )

    new_model = deepcopy(model)
    new_model.concept_model.concept_network = Chain(
        new_model.concept_model.concept_network, Randomize()
    )
    if hasattr(new_model.concept_model, "concept_prob_generators"):
        new_generators = nn.ModuleList()
        for generator in new_model.concept_model.concept_prob_generators:
            new_chain = Chain(generator, Randomize())
            new_generators.append(new_chain)
        new_model.concept_model.concept_prob_generators = new_generators
    results = test(new_model, test_loader)
    if dataset != "mimic_cxr":
        return results["test_acc"]
    else:
        return results["test_auroc"]
    # return results["test_acc"]


def test_random_residual(
    model: ConceptLightningModel, test_loader: DataLoader, dataset: str
) -> float:
    """
    Test model accuracy with randomized residual values.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    # Shuffle data
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=True,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory,
    )

    new_model = deepcopy(model)
    new_model.concept_model.residual_network = Chain(
        new_model.concept_model.residual_network, Randomize()
    )
    # self.concept_prob_generators = concept_network
    # self.concept_context_generators = residual_network
    if hasattr(new_model.concept_model, "concept_context_generators"):
        new_generators = nn.ModuleList()
        for generator in new_model.concept_model.concept_context_generators:
            new_chain = Chain(generator, Randomize())
            new_generators.append(new_chain)
        new_model.concept_model.concept_context_generators = new_generators
    results = test(new_model, test_loader)
    if dataset != "mimic_cxr":
        return results["test_acc"]
    else:
        return results["test_auroc"]


def test_correlation(model: ConceptLightningModel, test_loader: DataLoader) -> float:
    """
    Test mean absolute cross correlation between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    """
    correlations = []
    for (data, concepts), target in test_loader:
        with torch.no_grad():
            _, residual, _ = model(data, concepts=concepts)
        correlations.append(cross_correlation(concepts, residual).abs().mean().item())

    return np.mean(correlations)


def test_mutual_info(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    num_mi_epochs: int = 5,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_mi_epochs : int
        Number of epochs to train mutual information estimator
    """
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    # Get mutual information estimator
    (data, concepts), targets = next(iter(test_loader))
    data = data.to(model.device)
    concepts = concepts.to(model.device)
    _, residual, _ = model(data, concepts=concepts)
    concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
    mutual_info_estimator = MutualInformationLoss(residual_dim, concept_dim)
    mutual_info_estimator = mutual_info_estimator.to(model.device)
    # Learn mutual information estimator
    for epoch in range(num_mi_epochs):
        for (data, concepts), targets in test_loader:
            data = data.to(model.device)
            concepts = concepts.to(model.device)
            with torch.no_grad():
                _, residual, _ = model(data, concepts=concepts)
            mutual_info_estimator.step(residual, concepts)

    # Calculate mutual information
    mutual_infos = []
    for (data, concepts), target in test_loader:
        data = data.to(model.device)
        concepts = concepts.to(model.device)
        with torch.no_grad():
            _, residual, _ = model(data, concepts=concepts)
        mutual_infos.append(mutual_info_estimator(residual, concepts).item())

    return np.mean(mutual_infos)


def test_counterfactual_2(
    model: ConceptLightningModel,
    test_loader: DataLoader,
) -> float:
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    # Get mutual information estimator
    (data, concepts), targets = next(iter(test_loader))
    model.eval()
    data = data.to(model.device)
    concepts = concepts.to(model.device)
    _, residual, _ = model(data, concepts=concepts)
    if type(residual) == tuple:
        residual = residual[0]
    concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
    base_ds = test_loader.dataset
    predicate_matrix = base_ds.predicate_matrix
    class_to_idx = base_ds.animals_class_to_idx
    predicates_name_to_idx = {v: k for k, v in enumerate(base_ds.selected_predicates)}
    accuracy_polar_list = []
    accuracy_still_brown_list = []
    accuracy_brown_list = []

    for (data, concepts), target in test_loader:
        data = data.to(model.device)
        concepts = concepts.to(model.device)

        mask = target == class_to_idx["grizzly bear"]
        if mask.sum() > 0:
            data_masked = data[mask]
            concepts_masked = concepts[mask].clone()
            concepts_masked_int = concepts_masked.clone()
            target_polar = (
                (torch.ones_like(target[mask]) * class_to_idx["polar bear"])
                .to(target.dtype)
                .to(model.device)
            ).clone()
            target_grizzly = (
                (torch.ones_like(target[mask]) * class_to_idx["grizzly bear"])
                .to(target.dtype)
                .to(model.device)
            ).clone()
            with torch.no_grad():
                intervention_idxs = torch.ones_like(concepts_masked)
                # intervene on the white concept
                concepts_masked_int[:, predicates_name_to_idx["white"]] = 1
                _, residual, y_pred_polar = model(
                    data_masked,
                    concepts=concepts_masked_int,
                    intervention_idxs=intervention_idxs,
                )
                _, residual, y_pred_brown = model(
                    data_masked,
                    concepts=concepts_masked,
                    intervention_idxs=intervention_idxs,
                )
                # breakpoint()
                correct_pred_polar = (
                    (y_pred_polar.argmax(dim=-1) == target_polar).float().cpu().numpy()
                )
                accuracy_polar_list.append(correct_pred_polar)

                incorrect_still_brown = (
                    (y_pred_polar.argmax(dim=-1) == target_grizzly)
                    .float()
                    .cpu()
                    .numpy()
                )
                accuracy_still_brown_list.append(incorrect_still_brown)

                correct_pred_grizzly = (
                    (y_pred_brown.argmax(dim=-1) == target_grizzly)
                    .float()
                    .cpu()
                    .numpy()
                )
                accuracy_brown_list.append(correct_pred_grizzly)

    return [
        np.mean(np.concatenate(accuracy_polar_list)),
        np.mean(np.concatenate(accuracy_still_brown_list)),
        np.mean(np.concatenate(accuracy_brown_list)),
    ]


def test_counterfactual(
    model: ConceptLightningModel,
    test_loader: DataLoader,
) -> tuple:
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    # Get mutual information estimator
    (data, concepts), targets = next(iter(test_loader))
    model.eval()
    data = data.to(model.device)
    concepts = concepts.to(model.device)
    _, residual, _ = model(data, concepts=concepts)
    if type(residual) == tuple:
        residual = residual[0]
    concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
    base_ds = test_loader.dataset
    predicate_matrix = base_ds.predicate_matrix
    class_to_idx = base_ds.animals_class_to_idx
    predicates_name_to_idx = {v: k for k, v in enumerate(base_ds.selected_predicates)}

    # Lists to store original and predicted classes
    original_classes = []
    predicted_classes_after_intervention = []

    for (data, concepts), target in test_loader:
        data = data.to(model.device)
        concepts = concepts.to(model.device)

        mask = target == class_to_idx["grizzly bear"]
        if mask.sum() > 0:
            data_masked = data[mask]
            concepts_masked = concepts[mask].clone()
            concepts_masked_int = concepts_masked.clone()

            # Store original class (should all be grizzly bear based on the mask)
            # original_classes.extend([class_to_idx["grizzly bear"]] * mask.sum().item())

            with torch.no_grad():
                intervention_idxs = torch.ones_like(concepts_masked)
                _, _, y_pred_grizzly = model(
                    data_masked,
                    concepts=concepts_masked,
                    intervention_idxs=intervention_idxs,
                )
                # Store what the predictions were before intervention
                predictions = y_pred_grizzly.argmax(dim=-1).cpu().numpy().tolist()
                original_classes.extend(predictions)
                # intervene on the white concept
                concepts_masked_int[:, predicates_name_to_idx["white"]] = 1
                _, _, y_pred_polar = model(
                    data_masked,
                    concepts=concepts_masked_int,
                    intervention_idxs=intervention_idxs,
                )

                # Store what the predictions changed to after intervention
                predictions = y_pred_polar.argmax(dim=-1).cpu().numpy().tolist()
                predicted_classes_after_intervention.extend(predictions)

    # Return the lists of original classes and predicted classes after intervention
    return original_classes, predicted_classes_after_intervention


def analyze_residuals_with_pca(
    model: ConceptLightningModel,
    test_loader: DataLoader,
) -> tuple:
    """
    Collects residuals for grizzly bears and polar bears, reduces them to 2D using PCA,
    and returns two lists of 2D points for visualization.

    Args:
        model: The concept lightning model
        test_loader: DataLoader for test data

    Returns:
        tuple: (grizzly_bear_points, polar_bear_points) - Two lists of 2D points after PCA
    """
    import numpy as np
    from sklearn.decomposition import PCA

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )

    # Get class indices
    base_ds = test_loader.dataset
    class_to_idx = base_ds.animals_class_to_idx
    grizzly_idx = class_to_idx["grizzly bear"]
    polar_idx = class_to_idx["polar bear"]

    # Lists to store residuals for each class
    grizzly_residuals = []
    polar_residuals = []

    model.eval()
    with torch.no_grad():
        for (data, concepts), target in test_loader:
            # Create masks for each class
            grizzly_mask = target == grizzly_idx
            polar_mask = target == polar_idx

            # Skip batch if no grizzly or polar bears
            if not (grizzly_mask.any() or polar_mask.any()):
                continue

            # Get combined mask for both classes
            combined_mask = grizzly_mask | polar_mask

            # Skip if no relevant samples
            if not combined_mask.any():
                continue

            # Apply mask to data and concepts
            masked_data = data[combined_mask].to(model.device)
            masked_concepts = concepts[combined_mask].to(model.device)
            masked_target = target[combined_mask]

            # Get model outputs including residuals only for relevant samples
            _, residual, _ = model(masked_data, concepts=masked_concepts)
            if type(residual) == tuple:
                residual = residual[0]

            # Add residuals to appropriate lists
            for i, t in enumerate(masked_target):
                if t.item() == grizzly_idx:
                    grizzly_residuals.append(residual[i].cpu().numpy())
                elif t.item() == polar_idx:
                    polar_residuals.append(residual[i].cpu().numpy())

    # Convert lists to numpy arrays
    grizzly_residuals = np.array(grizzly_residuals)
    polar_residuals = np.array(polar_residuals)

    # Check if we have samples for both classes
    if len(grizzly_residuals) == 0:
        print("Warning: No grizzly bear samples found in the dataset")
        return [], []

    if len(polar_residuals) == 0:
        print("Warning: No polar bear samples found in the dataset")
        return [], []

    # Convert lists to numpy arrays
    grizzly_residuals = np.array(grizzly_residuals)
    polar_residuals = np.array(polar_residuals)

    # Combine all residuals for PCA fitting
    all_residuals = np.vstack([grizzly_residuals, polar_residuals])

    # Apply PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    pca.fit(all_residuals)

    # Transform each class's residuals to 2D
    grizzly_points = pca.transform(grizzly_residuals).tolist()
    polar_points = pca.transform(polar_residuals).tolist()

    return grizzly_points, polar_points


def test_confusion_matrix(
    model: ConceptLightningModel,
    test_loader: DataLoader,
) -> float:
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    # Get mutual information estimator
    # (data, concepts), targets = next(iter(test_loader))
    # data = data.to(model.device)
    # concepts = concepts.to(model.device)
    # _, residual, _ = model(data, concepts=concepts)
    # model.concept_model.training = False
    # concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
    # base_ds = test_loader.dataset
    from sklearn.metrics import confusion_matrix

    # Initialize lists to store true labels and predictions
    all_targets = []
    all_predictions = []
    model.eval()

    # Iterate through the test loader
    for (data, concepts), target in test_loader:
        data = data.to(model.device)
        concepts = concepts.to(model.device)
        target = target.to(model.device)

        with torch.no_grad():
            intervention_idxs = torch.ones_like(concepts)
            # intervene on the white concept
            _, residual, y_pred_polar = model(
                data,
                concepts=concepts,
                intervention_idxs=intervention_idxs,
            )
            # model.test_step(
            #     ((data, concepts), target), batch_idx=0, return_intervention_idxs=False
            # )

            # Get predicted class (assuming y_pred_polar contains class probabilities)
            _, predicted = torch.max(y_pred_polar, 1)

            # Append batch predictions and targets
            all_targets.append(target.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            # print(all_targets[0])
            # print(all_predictions[0])
            # print(y_pred_polar[0])

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    # new_model = deepcopy(model)
    # new_model.num_test_interventions = 6
    # new_model.concept_model.negative_intervention = False

    # new_model.concept_model.target_network = Chain(
    #     intervention,
    #     new_model.concept_model.target_network,
    # )
    # results = test(new_model, test_loader)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    return cm


def test_concept_pred(
    model: ConceptLightningModel,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_train_epochs: int = 5,
    dataset=None,
    data_dir=None,
    num_concepts=None,
    backbone=None,
    subset=None,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_train_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    if dataset == "celeba":
        hidden_concepts = 0
    else:
        hidden_concepts = 0

    concept_loss_fn = get_concept_loss_fn(
        dataset,
        data_dir,
        num_concepts=num_concepts,
        backbone=backbone,
        subset=subset,
    )
    print(concept_loss_fn)

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    (data, concepts), targets = next(iter(test_loader))
    data = data.to(model.device)
    concepts = concepts.to(model.device)

    if hidden_concepts != 0:
        _, residual, _ = model(data, concepts=concepts[:, :-hidden_concepts])
    else:
        _, residual, _ = model(data, concepts=concepts)

    if residual.shape[-1] < 1:
        print("Residual is empty")
        return [0, 0, 0, 0]

    if type(residual) == tuple:
        residual = residual[0]
    if model_type == "cem" or model_type == "cem_mi":
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1] // 2
        concept_predictor = ConceptEmbeddingConceptPred(
            residual_dim,
            concept_dim - hidden_concepts,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
            concept_loss_fn=concept_loss_fn,
        )
    else:
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
        concept_predictor = ConceptResidualConceptPred(
            residual_dim,
            concept_dim,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
            concept_loss_fn=concept_loss_fn,
        )
    concept_predictor = concept_predictor.to(model.device)

    best_val_loss = float("inf")
    best_predictor_state = None

    # Train the concept predictor
    for epoch in range(num_train_epochs):
        # Training phase
        concept_predictor.train()
        for (data, concepts), targets in train_loader:
            data = data.to(model.device)
            concepts = concepts.to(model.device)
            if model_type == "cem" or model_type == "cem_mi":
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        pre_contexts, residual, _ = model(data, concepts=concepts)
                if type(residual) == tuple:
                    residual = residual[0]
                contexts = pre_contexts.sigmoid()
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                concept_predictor.step(x.detach(), concepts.detach())
            else:
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        _, residual, _ = model(data, concepts=concepts)
                if type(residual) == tuple:
                    residual = residual[0]
                try:
                    concept_predictor.step(residual.detach(), concepts.detach())
                except Exception as e:
                    breakpoint()

        # Validation phase
        val_losses = []
        concept_predictor.eval()
        for (data, concepts), targets in val_loader:
            data = data.to(model.device)
            concepts = concepts.to(model.device)
            with torch.no_grad():
                if hidden_concepts != 0:
                    pre_contexts, residual, _ = model(
                        data, concepts=concepts[:, :-hidden_concepts]
                    )
                else:
                    pre_contexts, residual, _ = model(data, concepts=concepts)
                if type(residual) == tuple:
                    residual = residual[0]
                if model_type == "cem" or model_type == "cem_mi":
                    contexts = pre_contexts.sigmoid()
                    r_dim = residual.shape[-1]
                    pos_embedding = residual[:, :, : r_dim // 2]
                    neg_embedding = residual[:, :, r_dim // 2 :]
                    x = pos_embedding * torch.unsqueeze(
                        contexts, dim=-1
                    ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                else:
                    x = residual
                y_pred = concept_predictor(x)

                val_loss = concept_loss_fn(y_pred, concepts).item()
                val_losses.append(val_loss)

        mean_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}: Validation loss = {mean_val_loss}")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_predictor_state = concept_predictor.state_dict()

    # Load the best predictor state
    if best_predictor_state is not None:
        concept_predictor.load_state_dict(best_predictor_state)
    concept_predictor = concept_predictor.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Evaluate the concept predictor
    metrics = []
    for i in range(concept_dim):
        metrics.append([])
    loss = []
    intchange_metrics = []
    for i in range(concept_dim):
        intchange_metrics.append([])

    predictions = [[] for _ in range(concept_dim)]
    ground_truth = [[] for _ in range(concept_dim)]

    for (data, concepts), target in test_loader:
        data = data.to(model.device)
        concepts = concepts.to(model.device)
        with torch.no_grad():
            if hidden_concepts != 0:
                pre_contexts, residual, _ = model(
                    data, concepts=concepts[:, :-hidden_concepts]
                )
            else:
                pre_contexts, residual, _ = model(data, concepts=concepts)
            if type(residual) == tuple:
                residual = residual[0]
            if model_type == "cem" or model_type == "cem_mi":
                contexts = pre_contexts.sigmoid()
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x_test = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
            else:
                x_test = residual
            y_pred_base = concept_predictor(x_test)
            if model.concept_model.concept_type == "binary":
                y_pred_base = torch.sigmoid(y_pred_base)
                for i in range(concept_dim):
                    pred = (y_pred_base[:, i] > 0.5).float()
                    predictions[i].append(pred)
                    ground_truth[i].append(concepts[:, i])

                    accuracy = (pred == concepts[:, i]).float().mean().item()
                    metrics[i].append(accuracy)
            else:
                for i in range(concept_dim):
                    mse = (
                        ((y_pred_base[:, i] - concepts[:, i]) ** 2).mean().sqrt().item()
                    )
                    metrics[i].append(mse)
                loss.append(torch.nn.functional.mse_loss(y_pred_base, concepts).sqrt())

            # perform concept interventions with concept full concepts
            if model_type == "cem" or model_type == "cem_mi":
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                if hidden_concepts != 0:
                    x_test = pos_embedding * torch.unsqueeze(
                        concepts[:, :-hidden_concepts], dim=-1
                    ) + neg_embedding * torch.unsqueeze(
                        1 - concepts[:, :-hidden_concepts], dim=-1
                    )
                else:
                    x_test = pos_embedding * torch.unsqueeze(
                        concepts, dim=-1
                    ) + neg_embedding * torch.unsqueeze(1 - concepts, dim=-1)
            else:
                x_test = residual

            y_pred_intervention = concept_predictor(x_test)

            if model.concept_model.concept_type == "binary":
                y_pred_intervention = torch.sigmoid(y_pred_intervention)
                for i in range(concept_dim):
                    pred_intervene = (y_pred_intervention[:, i] > 0.5).float()
                    pred = (y_pred_base[:, i] > 0.5).float()
                    change = (pred != pred_intervene).float().mean().item()
                    intchange_metrics[i].append(change)

    # Calculate mean metric for each concept
    mean_metrics = np.array([np.mean(metric) for metric in metrics])
    mean_change_metrics = np.array([np.mean(metric) for metric in intchange_metrics])

    if model.concept_model.concept_type == "binary":
        predictions = [torch.cat(pred).to("cpu") for pred in predictions]
        ground_truth = [torch.cat(gt).to("cpu") for gt in ground_truth]
        from torchmetrics.classification import BinaryF1Score, BinaryAccuracy

        f1_scores = np.array(
            [
                BinaryF1Score()(pred, gt).item()
                for pred, gt in zip(predictions, ground_truth)
            ]
        )
    else:
        f1_scores = mean_metrics

    if hidden_concepts > 0:
        return np.array(
            [
                np.mean(f1_scores[:-hidden_concepts]),
                np.mean(f1_scores[-hidden_concepts:]),
                np.mean(mean_change_metrics[:-hidden_concepts]),
                np.mean(mean_change_metrics[-hidden_concepts:]),
            ]
        )
    else:
        return (
            np.mean(f1_scores),
            f1_scores,
            np.mean(mean_change_metrics),
            0,
        )


def test_concept_change_probe(
    model: ConceptLightningModel,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_train_epochs: int = 1,
    dataset=None,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_train_epochs : int
        Number of epochs to train mutual information estimator
    """
    # Get mutual information estimator
    if dataset == "celeba":
        hidden_concepts = 2
    else:
        hidden_concepts = 0
    (data, concepts), targets = next(iter(test_loader))
    if hidden_concepts != 0:
        _, residual, _ = model(data, concepts=concepts[:, :-hidden_concepts])
    else:
        _, residual, _ = model(data, concepts=concepts)
    if model_type == "cem" or model_type == "cem_mi":
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1] // 2
        concept_predictor = ConceptResidualConceptPred(
            (concept_dim - hidden_concepts) * residual_dim,
            concept_dim,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
        )
    else:
        concept_dim, residual_dim = concepts.shape[-1], residual.shape[-1]
        concept_predictor = ConceptResidualConceptPred(
            (concept_dim - hidden_concepts) + residual_dim,
            concept_dim,
            binary=model.concept_model.concept_type == "binary",
            hidden_concept=hidden_concepts > 0,
            num_hidden_concept=hidden_concepts,
        )

    best_val_loss = float("inf")
    best_predictor_state = None

    # Train the concept predictor
    for epoch in range(num_train_epochs):
        # Training phase
        concept_predictor.train()
        for (data, concepts), targets in train_loader:
            if model_type == "cem" or model_type == "cem_mi":
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        pre_contexts, residual, _ = model(data, concepts=concepts)
                contexts = pre_contexts.sigmoid()
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                x = x.reshape((x.shape[0], -1))
                concept_predictor.step(x.detach(), concepts.detach())
            else:
                with torch.no_grad():
                    if hidden_concepts != 0:
                        pre_contexts, residual, _ = model(
                            data, concepts=concepts[:, :-hidden_concepts]
                        )
                    else:
                        pre_contexts, residual, _ = model(data, concepts=concepts)
                x = torch.cat((residual, pre_contexts), dim=-1)
                concept_predictor.step(x.detach(), concepts.detach())

        # Validation phase
        val_losses = []
        concept_predictor.eval()
        for (data, concepts), targets in val_loader:
            with torch.no_grad():
                if hidden_concepts != 0:
                    pre_contexts, residual, _ = model(
                        data, concepts=concepts[:, :-hidden_concepts]
                    )
                else:
                    pre_contexts, residual, _ = model(data, concepts=concepts)
                if model_type == "cem" or model_type == "cem_mi":
                    contexts = pre_contexts.sigmoid()
                    r_dim = residual.shape[-1]
                    pos_embedding = residual[:, :, : r_dim // 2]
                    neg_embedding = residual[:, :, r_dim // 2 :]
                    x = pos_embedding * torch.unsqueeze(
                        contexts, dim=-1
                    ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                    x = x.reshape((x.shape[0], -1))
                else:
                    x = torch.cat((residual, pre_contexts), dim=-1)

                y_pred = concept_predictor(x)
                if model.concept_model.concept_type == "binary":
                    loss_fn = nn.BCEWithLogitsLoss()
                else:
                    loss_fn = nn.MSELoss()
                val_loss = loss_fn(y_pred, concepts).item()
                val_losses.append(val_loss)

        mean_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}: Validation loss = {mean_val_loss}")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_predictor_state = concept_predictor.state_dict()

    # Load the best predictor state
    if best_predictor_state is not None:
        concept_predictor.load_state_dict(best_predictor_state)

    # Evaluate the concept predictor
    metrics = []
    for i in range(concept_dim):
        metrics.append([])

    num_changed_concepts_list = []
    concept_updated_list = []
    hidden_concepts_updated_list = []

    for (data, concepts), target in test_loader:
        with torch.no_grad():
            if hidden_concepts != 0:
                model.num_test_interventions = 1
                tup, int_idxs = model.forward_intervention(
                    ((data, concepts[:, :-hidden_concepts]), target),
                    0,
                    return_intervention_idxs=True,
                )
                pre_contexts, residual, _ = tup
                contexts = pre_contexts.sigmoid()
                intervened_contexts = (
                    contexts.detach() * (1 - int_idxs)
                    + concepts[:, :-hidden_concepts] * int_idxs
                )
            else:
                model.num_test_interventions = 1
                tup, int_idxs = model.forward_intervention(
                    ((data, concepts), target), 0, return_intervention_idxs=True
                )
                pre_contexts, residual, _ = tup
                contexts = pre_contexts.sigmoid()
                intervened_contexts = (
                    contexts.detach() * (1 - int_idxs) + concepts * int_idxs
                )

            if model_type == "cem" or model_type == "cem_mi":
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]
                x_test = pos_embedding * torch.unsqueeze(
                    contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - contexts, dim=-1)
                x_test = x_test.reshape((x_test.shape[0], -1))
            else:
                x_test = torch.cat((residual, contexts), dim=-1)
            y_pred_base = concept_predictor(x_test)

            if model.concept_model.concept_type == "binary":
                y_pred_base = torch.sigmoid(y_pred_base)
                for i in range(concept_dim):
                    pred = (y_pred_base[:, i] > 0.5).float()
                    accuracy = (pred == concepts[:, i]).float().mean().item()
                    metrics[i].append(accuracy)
            else:
                for i in range(concept_dim):
                    mse = ((y_pred_base[:, i] - concepts[:, i]) ** 2).mean().item()
                    metrics[i].append(mse)

            # perform concept interventions with concept full concepts
            if model_type == "cem" or model_type == "cem_mi":
                r_dim = residual.shape[-1]
                pos_embedding = residual[:, :, : r_dim // 2]
                neg_embedding = residual[:, :, r_dim // 2 :]

                x_test = pos_embedding * torch.unsqueeze(
                    intervened_contexts, dim=-1
                ) + neg_embedding * torch.unsqueeze(1 - intervened_contexts, dim=-1)
                x_test = x_test.reshape((x_test.shape[0], -1))

            else:
                x_test = torch.cat((residual, intervened_contexts), dim=-1)

            y_pred_intervention = concept_predictor(x_test)
            if model.concept_model.concept_type == "binary":
                y_pred_intervention = torch.sigmoid(y_pred_intervention)

            pred_concepts = np.array(y_pred_base >= 0.5)
            pred_int_concepts = np.array(y_pred_intervention >= 0.5)
            np_concepts = np.array(concepts)

            # Vectorized calculations
            mask = np.array(
                int_idxs == 0
            )  # Assuming int_idxs is of shape (batch_size, 6)
            # mask = np.pad(
            #     mask, ((0, 0), (0, 2)), "constant", constant_values=0
            # )  # Add buffer of 0's to the right to make it (batch_size, 8)
            # assert (
            #     0
            # ), f"{int_idxs.shape} {pred_concepts.shape} {pred_int_concepts.shape} {concepts.shape}"
            if dataset == "celeba":
                num_changed_concepts = np.sum(
                    (pred_concepts[:, :6] != pred_int_concepts[:, :6]) & ~mask, axis=1
                )
                concept_updated = np.any(
                    np_concepts[:, :6] != pred_int_concepts[:, :6] & mask, axis=1
                )
                hidden_concepts_updated = np.sum(
                    pred_concepts[:, 6:8] != pred_int_concepts[:, 6:8], axis=1
                )
            else:
                num_changed_concepts = np.sum(
                    (pred_concepts != pred_int_concepts) & ~mask, axis=1
                )
                concept_updated = np.any(
                    np_concepts != pred_int_concepts & mask, axis=1
                )
                hidden_concepts_updated = [0.0]

            num_changed_concepts_list.extend(num_changed_concepts)
            concept_updated_list.extend(concept_updated)
            hidden_concepts_updated_list.extend(hidden_concepts_updated)

        # assert (
        #     0
        # ), f"{gt_concepts[9]} and {pred_concepts[9]} and {pred_int_concepts[9]} and {int_idxs.shape}"

    # Calculate mean metrics
    mean_accuracy = np.array([np.mean(metric) for metric in metrics])
    mean_num_changed_concepts = np.mean(num_changed_concepts_list)
    mean_concept_updated = np.mean(concept_updated_list)
    mean_hidden_concepts_updated = np.mean(hidden_concepts_updated_list)
    return (
        mean_accuracy,
        mean_num_changed_concepts,
        mean_concept_updated,
        mean_hidden_concepts_updated,
    )


def test_concept_change(
    model: ConceptLightningModel,
    model_type: str,
    test_loader: DataLoader,
    dataset=None,
    celeba=True,
) -> float:
    """
    Test mutual information between concepts and residuals.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    num_mi_epochs : int
        Number of epochs to train mutual information estimator
    """

    # Get mutual information estimator
    def invert_binarize(binary_int):
        binary_str = bin(binary_int)[2:].zfill(8)
        concepts = np.array([int(bit) for bit in binary_str], dtype=int)
        return concepts

    if dataset == "celeba":
        with open("/home/renos/label_invert.json", "r") as f:
            label_invert = json.load(f)
        (data, concepts), targets = next(iter(test_loader))
        _, residual, _ = model(data, concepts=concepts)

        def update_all(vector):
            return np.array(
                [invert_binarize(int(label_invert[str(int(v))])) for v in vector]
            )

    else:

        def update_all(vector):
            return np.array([invert_binarize(int(v)) for v in vector])

    num_changed_concepts_list = []
    concept_updated_list = []
    int_concept_correct_list = []
    base_concept_correct_list = []
    hidden_concepts_updated_list = []

    for (data, concepts), target in test_loader:
        with torch.no_grad():
            _, residual, y_pred = model(data, concepts=concepts)
            model.num_test_interventions = 1
            tup, int_idxs = model.forward_intervention(
                ((data, concepts), target), 0, return_intervention_idxs=True
            )
            _, _, y_pred_int = tup
        y_pred_amax = torch.argmax(y_pred, dim=1)
        y_pred_int_amex = torch.argmax(y_pred_int, dim=1)
        gt_concepts = update_all(target)
        pred_concepts = update_all(y_pred_amax)
        pred_int_concepts = update_all(y_pred_int_amex)

        # Vectorized calculations
        # only other concepts
        mask = np.array(int_idxs == 0)  # Assuming int_idxs is of shape (batch_size, 6)

        # number of supervised concepts changed during an intervention
        num_changed_concepts = np.sum(
            (pred_concepts[:, :6] != pred_int_concepts[:, :6]) & mask, axis=1
        )
        # Did an intervention change a concept?
        concept_updated = np.any(
            (pred_concepts[:, :6] != pred_int_concepts[:, :6]) & ~mask, axis=1
        )

        # Is concept correct after intervention?
        int_concept_correct = np.any(
            (gt_concepts[:, :6] == pred_int_concepts[:, :6]) & ~mask, axis=1
        )
        # Is concept correct before?
        base_concept_correct = np.any(
            (gt_concepts[:, :6] == pred_concepts[:, :6]) & ~mask, axis=1
        )
        hidden_concepts_updated = np.sum(
            pred_concepts[:, 6:8] != pred_int_concepts[:, 6:8], axis=1
        )

        concept_updated_list.extend(concept_updated)
        num_changed_concepts_list.extend(num_changed_concepts)
        int_concept_correct_list.extend(int_concept_correct)
        base_concept_correct_list.extend(base_concept_correct)
        hidden_concepts_updated_list.extend(hidden_concepts_updated)

        # assert (
        #     0
        # ), f"{gt_concepts[9]} and {pred_concepts[9]} and {pred_int_concepts[9]} and {int_idxs.shape}"
    num_changed_concepts = np.array(num_changed_concepts_list)
    concept_updated = np.array(concept_updated_list).astype(bool)

    int_concept_correct = np.array(int_concept_correct_list)
    base_concept_correct = np.array(base_concept_correct_list).astype(bool)
    hidden_concepts_updated = np.array(hidden_concepts_updated_list)
    base_concept_correct = np.array(base_concept_correct_list).astype(bool)

    # Calculate Metrics
    mean_num_changed_concepts = np.mean(num_changed_concepts)
    mean_hidden_concepts_updated = np.mean(hidden_concepts_updated)
    concept_updated_when_wrong = np.sum(
        concept_updated & ~base_concept_correct
    ) / np.sum(~base_concept_correct)

    return (
        mean_num_changed_concepts,
        concept_updated_when_wrong,
        mean_hidden_concepts_updated,
    )


def test_deep_lift_shapley(
    model: ConceptLightningModel,
    test_loader: DataLoader,
    dataset: str,
) -> dict:
    """
    Calculate DeepLift Shapley values for each concept and residual using Captum.
    Takes the absolute value of each sample before averaging to better capture
    the magnitude of impact regardless of direction.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    dataset : str
        Dataset name

    Returns
    -------
    dict
        Dictionary containing attribution scores for concepts and residuals
    """
    from captum.attr import DeepLift, DeepLiftShap
    import torch.nn.functional as F
    import torch

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )

    # Create a wrapper class for the concept model that Captum can use
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, concepts, residuals):
            # Combine concepts and residuals in the format expected by the model
            if not concepts.requires_grad:
                concepts = concepts.detach().requires_grad_()
            if not residuals.requires_grad:
                residuals = residuals.detach().requires_grad_()

            # Use the target_network from the concept model to get predictions
            return self.model.concept_model.calc_target_preds(
                concepts,
                residuals,
                concepts.detach(),
                torch.zeros_like(concepts).detach(),
                detach=False,
                get_concept_pred=False,
            )

    # Create the wrapped model
    wrapped_model = ModelWrapper(model)

    # Initialize DeepLift
    deep_lift = DeepLiftShap(wrapped_model)

    # Store attribution scores
    concept_attributions = []
    residual_attributions = []
    concept_values = []

    num_classes = DATASET_INFO[dataset]["num_classes"]
    max_samples = 10000
    device = model.device
    all_concept_inputs = []
    all_residual_outputs = []
    all_targets = []

    print("Phase 1: Collecting data for baseline calculation...")
    # First pass: collect all concept inputs and residual outputs
    num_processed = 0
    with torch.no_grad():
        for (data, concepts), targets in tqdm(test_loader):
            # Skip if we've processed enough samples
            if num_processed >= max_samples:
                break

            data = data.to(device)
            concepts = concepts.to(device)

            # Get model predictions and extract concepts and residuals
            concept_outputs, residual_outputs, _ = model(data, concepts=concepts)
            if type(residual_outputs) == tuple:
                residual_outputs = residual_outputs[0]

            # For binary concepts, ensure they're in the right format
            if (
                hasattr(model.concept_model, "concept_type")
                and model.concept_model.concept_type == "binary"
            ):
                concept_inputs = concepts
            else:
                concept_inputs = concepts

            # Store concepts and residuals for baseline calculation
            all_concept_inputs.append(concept_inputs.cpu())
            all_residual_outputs.append(residual_outputs.cpu())
            all_targets.append(targets.cpu())

            num_processed += len(data)

    # Calculate global means
    print("Calculating global mean baselines...")
    all_concepts = torch.cat(all_concept_inputs, dim=0)
    all_residuals = torch.cat(all_residual_outputs, dim=0)

    concept_mean = all_concepts.mean(dim=0, keepdim=False)
    residual_mean = all_residuals.mean(dim=0, keepdim=False)

    print(f"Concept mean shape: {concept_mean.shape}")
    print(f"Residual mean shape: {residual_mean.shape}")

    # Reset for second pass
    num_processed = 0

    print("Phase 2: Calculating attributions with global mean baselines...")
    # Second pass: calculate attributions using the global means
    for batch_idx in range(len(all_concept_inputs)):
        concept_inputs = all_concept_inputs[batch_idx].to(device)
        residual_outputs = all_residual_outputs[batch_idx].to(device)
        targets = all_targets[batch_idx].to(device)

        # Create baselines using the global means (expand to match batch size)
        if dataset == "oai":
            concept_baseline = concept_mean.expand(concept_inputs.shape[0], -1).to(
                device
            )
        else:
            # if concepts are binary, we want our baseline to be the uncertainty case where we're unsure of which class it is
            concept_baseline = (
                torch.zeros_like(concept_inputs).float().reshape(concept_inputs.shape)
            ) + 0.5
        residual_baseline = residual_mean.expand(residual_outputs.shape[0], -1).to(
            device
        )

        # Prepare inputs for DeepLift
        baselines = (concept_baseline, residual_baseline)
        inputs = (concept_inputs, residual_outputs)

        # Compute attributions
        attributions = deep_lift.attribute(inputs, baselines, target=targets)

        # Extract attributions and take absolute value for each sample
        concept_attr = attributions[0].cpu().detach().abs()
        residual_attr = attributions[1].cpu().detach().abs()

        concept_attributions.append(concept_attr)
        residual_attributions.append(residual_attr)
        concept_values.append(concept_inputs.cpu())
        num_processed += len(data)
    # Concatenate all samples' attributions
    all_concept_attr = torch.cat(concept_attributions, dim=0)
    all_residual_attr = torch.cat(residual_attributions, dim=0)
    all_concept_values = torch.cat(concept_values, dim=0)

    # Compute mean across all samples (after taking absolute values)
    avg_concept_attr = all_concept_attr  # torch.mean(all_concept_attr, dim=0)
    avg_residual_attr = all_residual_attr  # torch.mean(all_residual_attr, dim=0)

    # Convert to numpy for easier handling
    concept_attr_np = avg_concept_attr.cpu().numpy()
    residual_attr_np = avg_residual_attr.cpu().numpy()

    # Create result dictionary
    attribution_results = {
        "concept_attributions": concept_attr_np,
        "residual_attributions": residual_attr_np,
        "concept_values": all_concept_values.cpu().numpy(),
    }

    # For datasets with known concept names, add named attributions
    if dataset in DATASET_INFO:
        dataset_info = DATASET_INFO[dataset]
        if "concept_names" in dataset_info:
            concept_names = dataset_info["concept_names"]
            named_concept_attrs = {
                name: float(attr) for name, attr in zip(concept_names, concept_attr_np)
            }
            attribution_results["named_concept_attributions"] = named_concept_attrs

    return attribution_results


def test_tcav_captum(
    model: ConceptLightningModel,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset: str,
) -> dict:
    """
    Calculate TCAV (Testing with Concept Activation Vectors) scores using Captum.

    Parameters
    ----------
    model : ConceptLightningModel
        Model to evaluate
    val_loader : DataLoader
        Validation data loader for training concept classifiers
    test_loader : DataLoader
        Test data loader for calculating TCAV scores
    dataset : str
        Dataset name

    Returns
    -------
    dict
        Dictionary containing TCAV scores for concepts
    """
    import torch

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    import os
    import glob
    import torch
    from captum.concept._utils.data_iterator import (
        dataset_to_dataloader,
        CustomIterableDataset,
    )
    from captum.concept import Concept
    from captum.concept import TCAV
    from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
    from scipy.stats import ttest_ind

    def format_float(f):
        return float("{:.3f}".format(f) if abs(f) >= 0.0005 else "{:.3e}".format(f))

    def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
        score_list = []
        for concepts in experimental_sets:
            score_list.append(
                scores["-".join([str(c.id) for c in concepts])][score_layer][
                    score_type
                ][idx]
                .cpu()
                .numpy()
            )

        return score_list

    def get_pval(
        scores, experimental_sets, score_layer, score_type, alpha=0.05, print_ret=False
    ):
        P1 = assemble_scores(scores, experimental_sets, 0, score_layer, score_type)
        P2 = assemble_scores(scores, experimental_sets, 1, score_layer, score_type)

        if print_ret:
            print(
                "P1[mean, std]: ", format_float(np.mean(P1)), format_float(np.std(P1))
            )
            print(
                "P2[mean, std]: ", format_float(np.mean(P2)), format_float(np.std(P2))
            )

        _, pval = ttest_ind(P1, P2)

        if print_ret:
            print("p-values:", format_float(pval))

        if pval < alpha:  # alpha value is 0.05 or 5%
            relation = "Disjoint"
            if print_ret:
                print("Disjoint")
        else:
            relation = "Overlap"
            if print_ret:
                print("Overlap")

        return P1, P2, format_float(pval), relation

    def get_tensor_from_filename(filename):
        return torch.load(filename).to("cuda" if torch.cuda.is_available() else "cpu")

    def assemble_concept(name, id, concepts_path="/data/Datasets/tcav/celeba/"):
        concept_path = os.path.join(concepts_path, name) + "/"
        dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
        concept_iter = dataset_to_dataloader(dataset)

        return Concept(id=id, name=name, data_iter=concept_iter)

    if dataset == "celeba":
        concepts_path = "/data/Datasets/tcav/celeba/"
        concept_names = [
            "Attractive",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Smiling",
            "Wearing_Lipstick",
            "Heavy_Makeup",
            "Wavy_Hair",
        ]
    elif dataset == "cifar100":
        concepts_path = "/data/Datasets/tcav/cifar/"
        concept_names = DATASET_INFO[dataset]["class_names"]
    else:
        assert 0, "Dataset not supported for TCAV"

    named_concepts = concept_set = [
        assemble_concept(name, j, concepts_path=concepts_path)
        for j, name in enumerate(concept_names)
    ]
    random_concept_sets = [
        assemble_concept(
            f"random_{i-len(concept_names)}", i, concepts_path=concepts_path
        )
        for i in range(len(concept_names), len(concept_names) + 100)
    ]
    print(f"Random concepts: {[c for c in random_concept_sets]}")
    print(f"Named concepts: {[c for c in named_concepts]}")
    all_concept_sets = []
    flattened_concept_sets = []
    for named_concept in named_concepts:
        concept_sets = []
        for random_concept in random_concept_sets:
            concept_set = [named_concept, random_concept]
            concept_sets.append(concept_set)
        all_concept_sets.append(concept_sets)
        flattened_concept_sets.extend(concept_sets)

    from captum.attr import DeepLift, DeepLiftShap
    import torch.nn.functional as F

    # Create a wrapper class for the concept model that Captum can use
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, data):
            # breakpoint()
            # print(tupled.shape)
            # Combine concepts and residuals in the format expected by the model
            _, _, target_logits = self.model(data, concepts=None)
            print(data.device)
            assert (
                type(target_logits) == torch.Tensor
            ), f"Expected Tensor, got {type(target_logits)}"
            return target_logits

    # Create the wrapped model
    wrapped_model = ModelWrapper(model.concept_model)

    layers = ["model.concept_residual_concat"]

    mytcav = TCAV(
        model=wrapped_model,
        layers=layers,
        layer_attr_method=LayerIntegratedGradients(
            model, None, multiply_by_inputs=False
        ),
    )

    num_classes = DATASET_INFO[dataset]["num_classes"]

    # Process batches
    num_processed = 0
    max_samples = 10000  # Limit the number of samples for computational efficiency
    import time

    concept_results = {}

    # for batch in test_loader:
    #     # Skip if we've processed enough samples
    #     (data, concepts), targets = batch
    #     # breakpoint()
    #     data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #     targets = targets.to(
    #         torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     )
    #     # Create a mask where the last concept is true
    #     for i, (named_concept, concept_sets) in enumerate(
    #         zip(named_concepts, all_concept_sets)
    #     ):
    #         mask = concepts[:, i].bool()
    #         print(f"Number of samples in mask: {mask.sum()}")
    #         print(f"Starting interpretation for {named_concept.name} at {time.time()}")
    #         data_masked = data[mask]
    #         targets_masked = targets[mask]
    concept_data = {nc.name: {"data": None, "targets": None} for nc in named_concepts}
    MIN_EXAMPLES = 64

    # Collect examples across batches
    for batch in test_loader:
        (data, concepts), targets = batch
        data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        targets = targets.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Break if we have enough examples for all concepts
        if all(
            concept_data[nc.name]["data"] is not None
            and concept_data[nc.name]["data"].shape[0] >= MIN_EXAMPLES
            for nc in named_concepts
        ):
            break

        # Accumulate examples for each concept
        for i, named_concept in enumerate(named_concepts):
            mask = concepts[:, i].bool()
            if mask.sum() > 0:
                data_masked = data[mask]
                targets_masked = targets[mask]

                if concept_data[named_concept.name]["data"] is None:
                    concept_data[named_concept.name]["data"] = data_masked
                    concept_data[named_concept.name]["targets"] = targets_masked
                else:
                    concept_data[named_concept.name]["data"] = torch.cat(
                        [concept_data[named_concept.name]["data"], data_masked], dim=0
                    )
                    concept_data[named_concept.name]["targets"] = torch.cat(
                        [concept_data[named_concept.name]["targets"], targets_masked],
                        dim=0,
                    )

    # Run TCAV analysis for each concept
    for i, (named_concept, concept_sets) in enumerate(
        zip(named_concepts, all_concept_sets)
    ):
        print(f"Starting interpretation for {named_concept.name}")
        # Take only the first MIN_EXAMPLES
        data_masked = concept_data[named_concept.name]["data"][:MIN_EXAMPLES]
        targets_masked = concept_data[named_concept.name]["targets"][:MIN_EXAMPLES]

        assert (
            data_masked.shape[0] >= MIN_EXAMPLES
        ), f"Not enough examples for {named_concept.name}"
        assert (
            targets_masked.shape[0] >= MIN_EXAMPLES
        ), f"Not enough targets for {named_concept.name}"

        tcav_scores = mytcav.interpret(
            inputs=data_masked,
            experimental_sets=concept_sets,
            target=targets_masked,
            n_steps=50,
        )
        P1, P2, pval, relation = get_pval(
            tcav_scores, concept_sets, layers[0], score_type="sign_count"
        )
        concept_results[named_concept.name] = {}
        concept_results[named_concept.name]["sign_count"] = {
            "P1": np.array(P1).tolist(),
            "P2": np.array(P2).tolist(),
            "pval": pval,
            "relation": relation,
        }
        P1, P2, pval, relation = get_pval(
            tcav_scores, concept_sets, layers[0], score_type="magnitude"
        )
        concept_results[named_concept.name]["magnitude"] = {
            "P1": np.array(P1).tolist(),
            "P2": np.array(P2).tolist(),
            "pval": pval,
            "relation": relation,
        }

    return concept_results


### Loading & Execution


def filter_eval_configs(configs: list[dict]) -> list[dict]:
    """
    Filter evaluation configs.

    Parameters
    ----------
    configs : list[dict]
        List of evaluation configs
    """
    configs_to_keep = []
    for config in configs:
        if config["model_type"] == "concept_whitening":
            if config["eval_mode"].endswith("intervention"):
                print("Interventions not supported for concept whitening models")
                continue

        if config["model_type"] == "no_residual" or config["residual_dim"] == 0:
            if config["eval_mode"] in ("correlation", "mutual_info", "concept_pred"):
                print("Correlation / MI metrics not available for no-residual models")
                continue

        configs_to_keep.append(config)

    return configs_to_keep


def evaluate(config: dict):
    """
    Evaluate a trained model.

    Parameters
    ----------
    config : dict
        Evaluation configuration dictionary
    """
    metrics = {}
    # Get data loader
    if (
        # config["eval_mode"] == "concept_change_probe"
        # and (config["dataset"] == "celeba" or config["dataset"] == "pitfalls_synthetic") or
        config["eval_mode"] == "tcav"
        and config["dataset"] == "celeba"
    ):
        new_config = copy.deepcopy(config)
        new_config["num_concepts"] = 8
        new_config["batch_size"] = 256
        train_loader = make_datamodule(**new_config).train_dataloader()
        val_loader = make_datamodule(**new_config).val_dataloader()
        test_loader = make_datamodule(**new_config).test_dataloader()
    else:
        new_config = copy.deepcopy(config)
        if config["eval_mode"] == "tcav":
            new_config["batch_size"] = 256
        train_loader = make_datamodule(**new_config).train_dataloader()
        val_loader = make_datamodule(**new_config).val_dataloader()
        test_loader = make_datamodule(**new_config).test_dataloader()

    # Load model
    tuner = LightningTuner("val_acc", "max")
    model = tuner.load_model(make_concept_model, config["train_result"])
    if config["dataset"] == "mimic_cxr":
        dataset_info = DATASET_INFO[config["dataset"]][config["subset"]]
    else:
        dataset_info = DATASET_INFO[config["dataset"]]
    # Evaluate model
    if config["eval_mode"] == "accuracy":
        results = test(model, test_loader)
        if config["dataset"] != "mimic_cxr":
            keys = ["test_acc", "test_concept_acc"]
        else:
            keys = ["test_auroc", "test_concept_acc"]
        for key in keys:
            if key in results:
                metrics[key] = results[key]
    elif config["eval_mode"] == "neg_intervention":
        concept_dim = dataset_info["concept_dim"]
        metrics["neg_intervention_accs"] = test_interventions(
            model, test_loader, concept_dim, config["dataset"], negative=True
        )

    elif config["eval_mode"] == "pos_intervention":
        concept_dim = dataset_info["concept_dim"]
        metrics["pos_intervention_accs"] = test_interventions(
            model, test_loader, concept_dim, config["dataset"], negative=False
        )
    elif config["eval_mode"] == "threshold_fitting":
        metrics["threshold_fitting"] = test_threshold_fitting(
            model, test_loader, config["dataset"]
        )

    elif config["eval_mode"] == "random_concepts":
        metrics["random_concept_acc"] = test_random_concepts(
            model, test_loader, config["dataset"]
        )

    elif config["eval_mode"] == "random_residual":
        metrics["random_residual_acc"] = test_random_residual(
            model, test_loader, config["dataset"]
        )

    elif config["eval_mode"] == "correlation":
        metrics["mean_abs_cross_correlation"] = test_correlation(model, test_loader)

    elif config["eval_mode"] == "mutual_info":
        metrics["mutual_info"] = test_mutual_info(model, test_loader)

    elif config["eval_mode"] == "concept_pred":

        metrics["concept_pred"] = test_concept_pred(
            model,
            config["model_type"],
            train_loader,
            val_loader,
            test_loader,
            dataset=config["dataset"],
            data_dir=config["data_dir"],
            num_concepts=config.get("num_concepts", -1),
            backbone=config.get("backbone", "resnet34"),
            subset=config.get("subset", None),
        )
    elif config["eval_mode"] == "concept_change":
        metrics["concept_change"] = test_concept_change(
            model,
            config["model_type"],
            test_loader,
            dataset=config["dataset"],
        )
    elif config["eval_mode"] == "concept_change_probe":
        metrics["concept_change_probe"] = test_concept_change_probe(
            model,
            config["model_type"],
            train_loader,
            val_loader,
            test_loader,
            dataset=config["dataset"],
        )
    if config["eval_mode"] == "tcav":
        # Use Captum's TCAV implementation
        metrics["tcav_scores"] = test_tcav_captum(
            model, val_loader, test_loader, config["dataset"]
        )
    elif config["eval_mode"] == "deeplift_shapley":
        metrics["deeplift_shapley"] = test_deep_lift_shapley(
            model, test_loader, config["dataset"]
        )
    elif config["eval_mode"] == "test_counterfactual":
        metrics["test_counterfactual"] = test_counterfactual(model, test_loader)
    elif config["eval_mode"] == "test_confusion_matrix":
        metrics["test_confusion_matrix"] = test_confusion_matrix(model, test_loader)
    elif config["eval_mode"] == "test_counterfactual_2":
        metrics["test_counterfactual_2"] = test_counterfactual_2(model, test_loader)
    elif config["eval_mode"] == "pca":
        metrics["pca"] = analyze_residuals_with_pca(model, test_loader)

    # Report evaluation metrics
    ray.train.report(metrics)


if __name__ == "__main__":
    MODES = [
        # "accuracy",
        # "neg_intervention",
        # "pos_intervention",
        # "random_concepts",
        # "random_residual",
        # "correlation",
        # "mutual_info",
        #"concept_pred",
        # "concept_change",
        # "concept_change_probe",
        # "tcav",
        #"deeplift_shapley", 
        # "threshold_fitting",
        # "test_counterfactual",
        "test_counterfactual_2",
        # "test_confusion_matrix",
        # "pca",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        help="Experiment directory",
    )
    parser.add_argument("--mode", nargs="+", default=MODES, help="Evaluation modes")
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["dataset", "model_type"],
        help="Config keys to group by when selecting best trial results",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained models (instead of best trial per group)",
    )
    parser.add_argument(
        "--num-cpus", type=float, default=1, help="Number of CPUs to use (per model)"
    )
    parser.add_argument(
        "--num-gpus", type=float, default=1, help="Number of GPUs to use (per model)"
    )
    args = parser.parse_args()

    # Recursively search for 'tuner.pkl' file within the provided directory
    # If multiple are found, use the most recently modified one
    experiment_paths = Path(args.exp_dir).resolve().glob("**/train/tuner.pkl")
    experiment_path = sorted(experiment_paths, key=os.path.getmtime)[-1].parent.parent

    # Load train results
    train_folder = "train"
    print("Loading training results from", experiment_path / train_folder)
    tuner = LightningTuner.restore(experiment_path / train_folder)
    if args.all:
        results = tuner.get_results()
    else:
        results = [
            group.get_best_result()
            for group in tuner.get_results(groupby=args.groupby).values()
        ]

    # Create evaluation configs
    results = [result for result in results if result.config is not None]
    eval_configs = filter_eval_configs(
        [
            {
                **result.config["train_loop_config"],
                "train_result": result,
                "eval_mode": mode,
            }
            for result in results
            for mode in args.mode
        ]
    )

    # Get available resources
    if args.num_gpus < 1:
        set_cuda_visible_devices(available_memory_threshold=args.num_gpus)

    # Run evaluations
    eval_folder = "eval"
    tuner = tune.Tuner(
        tune.with_resources(
            evaluate,
            resources={
                "cpu": args.num_cpus,
                "gpu": args.num_gpus if torch.cuda.is_available() else 0,
            },
        ),
        param_space=tune.grid_search(eval_configs),
        run_config=air.RunConfig(name=eval_folder, storage_path=experiment_path),
    )
    eval_results = tuner.fit()
