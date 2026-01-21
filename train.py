"""
Training script for concept residual models.

This script provides the main entry point for training concept bottleneck models
with various residual disentanglement methods using Ray Tune for hyperparameter
search and distributed training.

Supported model types:
- no_residual: Standard CBM without residual
- latent_residual: CBM with unconstrained latent residual
- decorrelated_residual: CBM with cross-correlation penalty on residual
- mi_residual: CBM with mutual information minimization (D-CRM)
- adversarial_decorrelation: CBM with adversarial residual decorrelation
- cem: Concept Embedding Model
- cem_mi: CEM with MI minimization
- iter_norm: CBM with iterative normalization bottleneck
- layer_norm: CBM with layer normalization bottleneck
- concept_whitening: Concept whitening model

Usage:
    python train.py --config configs.cifar_baselines
    python train.py --config configs.celeba_mi_residual --data-dir ./data
"""

from __future__ import annotations

import torch

torch.set_float32_matmul_precision("high")

import argparse
import importlib
import pytorch_lightning as pl

from datetime import datetime
from pathlib import Path
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.nn import functional as F

from datasets import (
    get_concept_loss_fn,
    get_target_loss_weights,
    get_dummy_batch,
    get_datamodule,
    DATASET_INFO,
)
from lightning_ray import LightningTuner, parse_args_dynamic
from models import *
from utils import cross_correlation, RayConfig


def make_concept_model(**config) -> ConceptLightningModel:
    """
    Create a concept model based on configuration.

    This factory function instantiates the appropriate concept model architecture
    and wraps it in a PyTorch Lightning module for training.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - experiment_module_name : str
            Name of the experiment module (e.g. 'experiments.cifar')
        - model_type : str
            Model type (see module docstring for options)
        - training_mode : str
            One of {'independent', 'sequential', 'joint'}
        - dataset : str
            Dataset name
        - data_dir : str
            Path to data directory
        - concept_dim : int
            Size of concept vector
        - residual_dim : int
            Size of residual vector
        - mi_estimator_hidden_dim : int
            Hidden dimension of MI estimator (for mi_residual)
        - mi_optimizer_lr : float
            Learning rate of MI estimator optimizer
        - cw_alignment_frequency : int
            Frequency of concept alignment for whitening (in epochs)

    Returns
    -------
    ConceptLightningModel
        Configured concept model wrapped in Lightning module
    """
    experiment_module = importlib.import_module(config["experiment_module_name"])
    model_type = config.get("model_type", "latent_residual")

    # Update config with dataset information (e.g. concept_dim, num_classes)
    if config.get("num_concepts", -1) != -1:
        DATASET_INFO[config["dataset"]]["concept_dim"] = config["num_concepts"]
        config["concept_dim"] = config["num_concepts"]

    if config["dataset"] == "mimic_cxr":
        dataset_info = DATASET_INFO[config["dataset"]][config["subset"]]
    else:
        dataset_info = DATASET_INFO[config["dataset"]]
    config = {**dataset_info, **config}

    # Configure loss functions
    config["concept_loss_fn"] = get_concept_loss_fn(
        config["dataset"],
        config["data_dir"],
        num_concepts=config.get("num_concepts", -1),
        backbone=config.get("backbone", "resnet34"),
        subset=config.get("subset", None),
    )
    if config["dataset"] == "mimic_cxr":
        config["target_loss_fn"] = get_target_loss_weights(
            config["dataset"],
            config["data_dir"],
            num_concepts=config.get("num_concepts", -1),
            backbone=config.get("backbone", "resnet34"),
            subset=config.get("subset", None),
        )
    else:
        config["target_loss_fn"] = F.cross_entropy

    # Instantiate model based on type
    if model_type == "no_residual":
        config = {**config, "residual_dim": 0}
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    elif model_type == "latent_residual":
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    elif model_type == "decorrelated_residual":
        residual_loss_fn = lambda r, c: cross_correlation(r, c).square().mean()
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(
            model, residual_loss_fn=residual_loss_fn, **config
        )

    elif model_type == "mi_residual":
        model = experiment_module.make_concept_model(config)
        model = MutualInfoConceptLightningModel(model, **config)

    elif model_type == "mi_residual_info_bottleneck":
        model = experiment_module.make_concept_model(config)
        model = MutualInfoConceptLightningModel(model, **config)

    elif model_type == "adversarial_decorrelation":
        model = experiment_module.make_concept_model(config)
        model = AdversarialDecorrelationConceptLightningModel(model, **config)

    elif model_type == "cem":
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    elif model_type == "cem_mi":
        model = experiment_module.make_concept_model(config)
        model = MutualInfoConceptLightningModel(model, concept_embedding=True, **config)

    elif model_type in ("iter_norm", "layer_norm"):
        config = {**config, "norm_type": model_type}
        model = experiment_module.make_concept_model(config)
        model = ConceptLightningModel(model, **config)

    elif model_type == "concept_whitening":
        config = {
            **config,
            "concept_type": "continuous",
            "norm_type": "concept_whitening",
            "training_mode": "joint",
        }
        model = experiment_module.make_concept_model(config)
        model = ConceptWhiteningLightningModel(model, **config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Dummy forward pass to initialize lazy layers
    batch = get_dummy_batch(
        config["dataset"],
        config["data_dir"],
        config.get("num_concepts", -1),
        config.get("backbone", "resnet34"),
        config.get("subset", None),
    )
    model.dummy_pass([batch])

    return model


def make_datamodule(**config) -> pl.LightningDataModule:
    """
    Create a data module for training.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing dataset parameters

    Returns
    -------
    pl.LightningDataModule
        Configured data module
    """
    return get_datamodule(
        dataset_name=config["dataset"],
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=int(config.get("num_cpus", 1)) - 1,
        resize_oai=config.get("resize_oai", True),
        num_concepts=config.get("num_concepts", -1),
        backbone=config.get("backbone", "resnet34"),
        subset=config.get("subset", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train concept residual models with Ray Tune"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments.cifar",
        help="Experiment configuration module (e.g., configs.cifar_baselines)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory where data is stored"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save models to"
    )
    parser.add_argument(
        "--restore-path",
        type=str,
        help="Path to restore model from (for resuming training)"
    )
    parser.add_argument(
        "--groupby",
        type=str,
        nargs="+",
        help="Config keys to group by for result analysis"
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Use AsyncHyperBand trial scheduler"
    )

    args, args_config = parse_args_dynamic(parser)

    # Load experiment config
    experiment_module = importlib.import_module(args.config)
    config = RayConfig(experiment_module.get_config())
    config.update({k: v for k, v in vars(args).items() if v is not None})
    config.update(args_config)
    config.set("experiment_module_name", args.config)
    config.set("data_dir", Path(config.get("data_dir")).expanduser().resolve())
    config.set("save_dir", Path(config.get("save_dir")).expanduser().resolve())

    # Pre-download datasets before launching Ray Tune workers
    # Avoids race conditions when multiple workers try to download simultaneously
    dataset_names = config.get("dataset")
    if isinstance(dataset_names, dict) and "grid_search" in dataset_names:
        dataset_names = list(dataset_names.values())
    dataset_names = [dataset_names] if isinstance(dataset_names, str) else dataset_names

    for dataset_name in dataset_names:
        get_datamodule(
            dataset_name,
            data_dir=config.get("data_dir"),
            backbone=config.get("backbone", "resnet34"),
            subset="cardiomegaly",
        )

    # Create trial scheduler (optional)
    scheduler = None
    if args.scheduler:
        scheduler = AsyncHyperBandScheduler(
            max_t=config.get("num_epochs"),
            grace_period=config.get("num_epochs") // 5
        )

    # Generate experiment name with timestamp
    date = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_name = config.get("experiment_module_name").split(".")[-1]
    experiment_name = f"{experiment_name}/{date}/train"

    # Configure metric for hyperparameter optimization
    config.set("max_epochs", config.get("num_epochs"))
    if args.restore_path:
        tuner = LightningTuner.restore(args.restore_path, resume_errored=True)
    else:
        if config.get("dataset") == "mimic_cxr":
            metric_to_max = "val_intervention_auroc"
        elif config.get("dataset") == "cub":
            metric_to_max = "val_acc"
        else:
            metric_to_max = "val_intervention_acc"

        tuner = LightningTuner(
            metric=metric_to_max,
            mode="max",
            scheduler=scheduler,
            num_samples=config.get("num_samples", 1),
        )

    # Run training
    tuner.fit(
        make_concept_model,
        make_datamodule,
        param_space=config,
        save_dir=args.save_dir or config.get("save_dir"),
        experiment_name=experiment_name,
        num_workers_per_trial=config.get("num_workers", 1),
        num_cpus_per_worker=config.get("num_cpus", 1),
        num_gpus_per_worker=config.get("num_gpus", 1),
        gpu_memory_per_worker=config.get("gpu_memory_per_worker", None),
        groupby=config.get("groupby", []),
    )
