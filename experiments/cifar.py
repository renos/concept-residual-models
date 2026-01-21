import os
from models.partial_prob_crm import PartialProbabilisticConceptModel
import ray
import torch.nn as nn
import torch

from models import (
    ConceptModel,
    ConceptEmbeddingModel,
    make_bottleneck_layer,
)
from nn_extensions import Apply
from utils import (
    make_cnn,
    make_mlp,
    process_grid_search_tuples,
    make_concept_embedding_model,
)
from .celeba import CrossAttentionModel, PassThrough


def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config["num_classes"]
    concept_dim = config["concept_dim"]
    residual_dim = config["residual_dim"]
    int_model_use_bn = config.get("int_model_use_bn", True)
    int_model_layers = config.get("int_model_layers", None)
    backbone = config.get("backbone", "resnet34")
    load_weights = config.get("load_weights", True)
    blackbox = config.get("blackbox", False)

    if config.get("model_type") == "cem" or config.get("model_type") == "cem_mi":
        bottleneck_dim = (
            concept_dim * residual_dim
        )  # residual dim is the size of the concept embedding for cem
    # elif config.get("additive_residual", False):
    #     bottleneck_dim = concept_dim
    elif (
        config.get("model_type") == "mi_residual_prob"
        or config.get("model_type") == "latent_residual_prob"
    ):
        bottleneck_dim = 2 * concept_dim + 2 * residual_dim
    else:
        bottleneck_dim = concept_dim + residual_dim

    if config.get("model_type") == "cem" or config.get("model_type") == "cem_mi":
        units = (
            [
                concept_dim * residual_dim + concept_dim
            ]  # for cem, input is concept_dim * residual_dim (# of concepts * concept embedding dim)
            + (int_model_layers or [256, 128])  # + previous interventions
            + [concept_dim]
        )
    elif config.get("additive_residual", False):
        # residual is added to concept not concatenated
        units = (
            [concept_dim + residual_dim]  # Bottleneck  # Prev interventions
            + (int_model_layers or [256, 128])
            + [concept_dim]
        )
    elif (
        config.get("model_type") == "mi_residual_prob"
        or config.get("model_type") == "latent_residual_prob"
    ):
        units = (
            [2 * concept_dim + 2 * residual_dim]  # Bottleneck  # Prev interventions
            + (int_model_layers or [256, 128])
            + [concept_dim]
        )
    else:
        units = (
            [
                concept_dim + residual_dim + concept_dim
            ]  # Bottleneck  # Prev interventions
            + (int_model_layers or [256, 128])
            + [concept_dim]
        )
    layers = []
    for i in range(1, len(units)):
        if int_model_use_bn:
            layers.append(
                torch.nn.BatchNorm1d(num_features=units[i - 1]),
            )
        layers.append(torch.nn.Linear(units[i - 1], units[i]))
        if i != len(units) - 1:
            layers.append(torch.nn.LeakyReLU())
    if config.get("use_batch_norm_layer", False):
        batch_norm_layer = nn.BatchNorm1d(num_features=bottleneck_dim)
    else:
        batch_norm_layer = None

    if config.get("intervention_weight", 0.0) > 0:
        concept_rank_model = torch.nn.Sequential(*layers)
    else:
        concept_rank_model = nn.Identity()
    if config.get("num_target_network_layers", 0):
        target_network = make_mlp(
            num_classes,
            num_hidden_layers=config.get("num_target_network_layers", 0),
            hidden_dim=16,
        )
    elif config.get("additive_residual", False):
        target_network = nn.Linear(concept_dim, num_classes)
    else:
        target_network = nn.Linear(bottleneck_dim, num_classes)
    if config.get("model_type") == "cem" or config.get("model_type") == "cem_mi":
        concept_prob_generators, concept_context_generators = (
            make_concept_embedding_model(
                1000, residual_dim, concept_dim, embedding_activation="leakyrelu"
            )
        )
        return ConceptEmbeddingModel(
            base_network=make_cnn(1000, cnn_type=backbone, load_weights=load_weights),
            concept_network=concept_prob_generators,
            residual_network=concept_context_generators,
            target_network=target_network,
            bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
            concept_rank_model=concept_rank_model,
            **config,
        )
    else:
        if config.get("cross", False) and residual_dim >= 4:
            cross_attention = CrossAttentionModel(
                concept_dim, residual_dim, residual_dim, min(residual_dim, 8)
            )
        else:
            cross_attention = PassThrough(concept_dim, residual_dim, residual_dim, 8)

        if config.get("model_type") == "mi_residual_info_bottleneck":
            return PartialProbabilisticConceptModel(
                base_network=make_cnn(
                    2 * residual_dim + concept_dim,
                    cnn_type=backbone,
                    load_weights=True,
                ),
                concept_network=Apply(lambda x: x[..., :concept_dim]),
                residual_network=Apply(lambda x: x[..., concept_dim:]),
                target_network=target_network,
                bottleneck_layer=make_bottleneck_layer(
                    residual_dim + concept_dim, **config
                ),
                cross_attention=cross_attention,
                concept_rank_model=concept_rank_model,
                **config,
            )
        return ConceptModel(
            base_network=make_cnn(
                bottleneck_dim, cnn_type=backbone, load_weights=load_weights
            ),
            concept_network=Apply(lambda x: x[..., :concept_dim]),
            residual_network=Apply(lambda x: x[..., concept_dim:]),
            target_network=target_network,
            bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
            cross_attention=cross_attention,
            concept_rank_model=concept_rank_model,
            batch_norm_layer=batch_norm_layer,
            **config,
        )


def get_config(**kwargs) -> dict:
    config = {
        # ("model_type", "beta"): ray.tune.grid_search(
        #     [
        #         ("latent_residual", 0),
        #         ("decorrelated_residual", 10.0),
        #         ("iter_norm", 0),
        #         ("mi_residual", 1.0),
        #     ]
        # ),
        "residual_dim": ray.tune.grid_search([0, 1, 2, 4, 8, 16, 32]),
        "dataset": "cifar100",
        "data_dir": os.environ.get("CONCEPT_DATA_DIR", "./data"),
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved"),
        "training_mode": "independent",
        "num_epochs": 100,
        "lr": 1e-4,
        "batch_size": 64,
        "alpha": 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 1e-4,
        "cw_alignment_frequency": 20,
        "checkpoint_frequency": 5,
        "gpu_memory_per_worker": "2500 MiB",
    }
    config.update(kwargs)
    config = process_grid_search_tuples(config)

    return config
