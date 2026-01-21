import os
import ray

from utils import process_grid_search_tuples

import os
import ray
import torch.nn as nn

from models import ConceptModel, ConceptEmbeddingModel, make_bottleneck_layer
from nn_extensions import Apply
from utils import (
    make_cnn,
    make_concept_embedding_model,
    make_mlp,
    process_grid_search_tuples,
)

from .celeba import CrossAttentionModel


def make_concept_model(config: dict) -> ConceptModel:
    num_classes = config["num_classes"]
    concept_dim = config["concept_dim"]
    residual_dim = config["residual_dim"]
    int_model_use_bn = config.get("int_model_use_bn", True)
    int_model_layers = config.get("int_model_layers", None)
    backbone = config.get("backbone", "resnet18")

    if config["model_type"].startswith("cem"):
        bottleneck_dim = concept_dim * residual_dim
        units = [
            concept_dim * residual_dim + concept_dim,
            *(int_model_layers or [256, 128]),
            concept_dim,
        ]

    else:
        bottleneck_dim = concept_dim + residual_dim
        units = [
            concept_dim + residual_dim + concept_dim,
            *(int_model_layers or [256, 128]),
            concept_dim,
        ]

    if config.get("intervention_weight", 0.0) > 0:
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(nn.BatchNorm1d(num_features=units[i - 1]))
            layers.append(nn.Linear(units[i - 1], units[i]))
            if i != len(units) - 1:
                layers.append(nn.LeakyReLU())

        concept_rank_model = nn.Sequential(*layers)

    else:
        concept_rank_model = nn.Identity()

    cross_attention = None
    if config["model_type"].startswith("cem"):
        model_cls = ConceptEmbeddingModel
        base_network = make_cnn(1000, cnn_type=backbone)
        concept_network, residual_network = make_concept_embedding_model(
            1000, residual_dim, concept_dim, embedding_activation="leakyrelu"
        )
    else:
        model_cls = ConceptModel
        if config.get("separate_branches", False):
            base_network = nn.Identity()
            concept_network = make_cnn(bottleneck_dim, cnn_type=backbone)
            residual_network = make_cnn(bottleneck_dim, cnn_type=backbone)
        else:
            base_network = make_cnn(bottleneck_dim, cnn_type=backbone)
            concept_network = Apply(lambda x: x[..., :concept_dim])
            residual_network = Apply(lambda x: x[..., concept_dim:])
        if config.get("cross", False) and residual_dim >= 4:
            cross_attention = CrossAttentionModel(
                concept_dim, residual_dim, residual_dim, min(residual_dim, 8)
            )

    target_network = make_mlp(
        num_classes,
        num_hidden_layers=2,
        hidden_dim=50,
    )
    config["intervention_aware"] = False
    return model_cls(
        base_network=base_network,
        concept_network=concept_network,
        residual_network=residual_network,
        target_network=target_network,
        bottleneck_layer=make_bottleneck_layer(bottleneck_dim, **config),
        concept_rank_model=concept_rank_model,
        cross_attention=cross_attention,
        **config,
    )



def get_config(**kwargs) -> dict:
    config = {
        # 'model_type': ray.tune.grid_search([
        #     'latent_residual',
        #     'decorrelated_residual',
        #     'iter_norm',
        #     'mi_residual',
        # ]),
        'residual_dim': ray.tune.grid_search([0, 1, 2, 4, 8, 16, 32, 64]),
        'dataset': 'oai',
        'data_dir': os.environ.get('CONCEPT_DATA_DIR', './data'),
        'save_dir': os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        'num_cpus': 4,
        'gpu_memory_per_worker': '5000 MiB',
        'training_mode': 'independent',
        'num_epochs': 100,
        'lr': 1e-4,
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-3,
        'cw_alignment_frequency': 20,
        'checkpoint_frequency': 5,
        "backbone": "resnet18",
        "num_target_network_layers": 2,
    }
    config.update(kwargs)
    config = process_grid_search_tuples(config)
    return config
