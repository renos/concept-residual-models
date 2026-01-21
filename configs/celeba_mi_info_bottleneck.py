import os
from ray import tune
from experiments.celeba import (
    get_config as get_celeba_config,
    make_concept_model,
)


def get_config(**kwargs) -> dict:
    experiment_config = {
        "model_type": "mi_residual_info_bottleneck",
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved/"),
        "data_dir": "/data/Datasets/celeba/",
        "ray_storage_dir": os.environ.get("RAY_STORAGE_DIR", "./ray_results/"),
        "residual_dim": 16,
        "lr": 0.001,
        "num_epochs": 50,
        "momentum": 0.9,
        # "lr_scheduler": "reduce_on_plateau",
        # "chosen_optim": "sgd",
        "lr_scheduler": "cosine annealing",
        "chosen_optim": "adam",
        "alpha": 1.0,
        "beta": tune.grid_search([0.0, 2.0]),
        "delta": 1e-3,
        "mi_const": tune.grid_search([0.25, 0.5, 1.0, 1.5]),
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 1,
        "num_gpus": 1.0,
        "num_samples": 5,
        "batch_size": 64,
        "checkpoint_frequency": 1,
        "norm_type": None,
        "T_whitening": 3,
        "weight_decay": 4e-6,
        "training_mode": "sequential",
        "complete_intervention_weight": 1.0,
        "training_intervention_prob": 0.25,
        "intervention_task_loss_weight": 1.0,
        "intervention_weight": 5.0,
        "patience": 3,
        "cross": False,
        "intervention_aware": False,  # True for intervention-aware training
        "num_target_network_layers": 0,  # Number of layers in the target network
        "weight_pred": False,
    }
    experiment_config.update(kwargs)
    experiment_config = get_celeba_config(**experiment_config)
    return experiment_config
