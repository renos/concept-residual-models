import os
from ray import tune
from experiments.cub import get_config as get_cub_config, make_concept_model


def get_config(**kwargs) -> dict:
    experiment_config = {
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved/"),
        "model_type": "mi_residual_info_bottleneck",
        "residual_dim": 32,
        "lr": 1e-4,
        "num_epochs": 300,
        "momentum": 0.9,
        # "lr_scheduler": "reduce_on_plateau",
        # "chosen_optim": "sgd",
        "lr_scheduler": "cosine annealing",
        "chosen_optim": "adam",
        "alpha": 1.0,
        "beta": tune.grid_search([0.0, 1.0]),
        "delta": 1e-3,
        "mi_const": tune.grid_search([0.5, 1.0, 1.5, 2.5, 3.0]),
        # "initial_horizon": 10,
        "max_horizon": 4,
        "mi_estimator_hidden_dim": 512,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 8,
        "num_gpus": 1.0,
        "num_samples": 5,
        "batch_size": 64,
        "checkpoint_frequency": 5,
        "norm_type": None,
        "T_whitening": 3,
        "weight_decay": 4e-6,
        "training_mode": "sequential",
        "num_hidden": 0,
        "complete_intervention_weight": 1.0,  # tune.grid_search([0.01, 0.1, 0.25, 0.5]),
        "training_intervention_prob": 0.25,
        "intervention_task_loss_weight": 0.0,
        "intervention_weight": 5.0,
        "intervention_aware": False,  # True for intervention-aware training
        "gpu_memory_per_worker": "14000 MiB",
        "cross": False,
        "backbone": "inception_v3",
    }
    experiment_config.update(kwargs)
    experiment_config = get_cub_config(**experiment_config)
    return experiment_config
