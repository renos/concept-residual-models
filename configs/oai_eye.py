import os
from ray import tune
from experiments.oai import (
    get_config as get_oai_config,
    make_concept_model,
)


def get_config(**kwargs) -> dict:
    experiment_config = {
        "model_type": "latent_residual",  # tune.grid_search(["ccm_eye", "ccm_r"]),
        "save_dir": os.environ.get("CONCEPT_SAVE_DIR", "./saved/"),
        "data_dir": os.environ.get("CONCEPT_DATA_DIR", "./data/"),
        "ray_storage_dir": os.environ.get("RAY_STORAGE_DIR", "./ray_results/"),
        "residual_dim": tune.grid_search([1, 2, 4, 8, 16, 32, 64, 128]),
        "lr": 1e-4,
        "num_epochs": 100,
        "alpha": 1.0,
        "reg_gamma": 0.01,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr": 0.001,
        "cw_alignment_frequency": 20,
        "num_cpus": 8,
        "num_gpus": 1.0,
        "num_samples": 5,
        "batch_size": 64,
        "checkpoint_frequency": 5,
        "reg_type": "eye",
        "T_whitening": 3,
        "complete_intervention_weight": 1.0,
        "training_intervention_prob": 0.25,
        "intervention_task_loss_weight": 0.0,
        "intervention_weight": 0.0,
        "intervention_aware": False,  # True for intervention-aware training
    }
    experiment_config.update(kwargs)
    experiment_config = get_oai_config(**experiment_config)
    return experiment_config
