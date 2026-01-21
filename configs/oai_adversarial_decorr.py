import os
from ray import tune
from experiments.oai import get_config as get_oai_config, make_concept_model


def get_config(**kwargs) -> dict:
    experiment_config = {
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved/"),
        "data_dir": os.environ.get("CONCEPT_DATA_DIR", "./data/"),
        "ray_storage_dir": os.environ.get("RAY_STORAGE_DIR", "./ray_results/"),
        "model_type": "adversarial_decorrelation",
        "residual_dim": tune.grid_search([1, 2, 4, 8, 16, 32, 64, 128]),
        "lr": 1e-4,
        "num_epochs": 100,
        "momentum": 0.9,
        "lr_scheduler": "cosine annealing",
        "chosen_optim": "adam",
        "alpha": 1.0,
        "beta": 1.0,
        "max_horizon": 4,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 0.001,
        "adv_decorr_hidden_dim": 256,  # Hidden dim for adversarial discriminator
        "adv_decorr_optimizer_lr": 0.001,  # Learning rate for discriminator
        "adv_decorr_num_steps": 3,  # Train discriminator 3 times per batch
        "lambda_adv": 10,  # Adversarial decorrelation strength
        "adv_decorr_frequency": 1,  # How often to update discriminator (every N batches)
        "max_error": 5.0,  # Maximum MSE clamp value for OAI dataset
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
        "complete_intervention_weight": 1.0,
        "training_intervention_prob": 0.25,
        "intervention_task_loss_weight": 0.0,
        "intervention_weight": 5.0,
        "intervention_aware": False,  # True for intervention-aware training
        "gpu_memory_per_worker": "14000 MiB",
        "cross": False,
        "backbone": "resnet18",
    }
    experiment_config.update(kwargs)
    experiment_config = get_oai_config(**experiment_config)
    return experiment_config
