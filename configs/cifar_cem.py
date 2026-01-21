import os
from ray import tune
from experiments.cifar import (
    get_config as get_cifar_config,
    make_concept_model,
)


def get_config(**kwargs) -> dict:
    experiment_config = {
        "model_type": "cem",
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved/"),
        "data_dir": os.environ.get("CONCEPT_DATA_DIR", "./data/"),
        "ray_storage_dir": os.environ.get("RAY_STORAGE_DIR", "./ray_results/"),
        "residual_dim": tune.grid_search([1, 2, 4, 8, 16]),
        "lr": 0.01,
        "num_epochs": 300,
        "momentum": 0.9,
        "lr_scheduler": "reduce_on_plateau",
        "chosen_optim": "sgd",
        "alpha": 1.0,
        "beta": 1.0,
        "max_horizon": 6,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 8,
        "num_gpus": 1.0,
        "num_samples": 5,
        "batch_size": 256,
        "checkpoint_frequency": 5,
        "norm_type": None,
        "T_whitening": 3,
        "weight_decay": 4e-6,
        "training_mode": "sequential",
        "num_hidden": 0,
        "complete_intervention_weight": 0.0,
        "training_intervention_prob": 0.25,
        "intervention_task_loss_weight": 1.0,
        "intervention_weight": 2.0,
        "backbone": "resnet18",
        "load_weights": False,
        "patience": 10,
    }
    experiment_config.update(kwargs)
    experiment_config = get_cifar_config(**experiment_config)
    return experiment_config
