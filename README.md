# Disentangled Concept-Residual Models (D-CRM)

Official implementation of **"Disentangled Concept-Residual Models: Bridging the Interpretability–Performance Gap for Incomplete Concept Sets"**, published in **Transactions on Machine Learning Research (TMLR), January 2026**.

[![TMLR](https://img.shields.io/badge/TMLR-2026-blue)](https://openreview.net/forum?id=NKgNizwDa6)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Abstract

Concept Bottleneck Models (CBMs) enhance interpretability by predicting human-understandable concepts before making final predictions. Concept Residual Models (CRMs) extend CBMs by adding a residual pathway to capture information not represented by the predefined concepts. However, we identify a critical issue: **concept-residual overlap**, where residuals redundantly encode concept information, undermining the interpretability benefits of concept interventions.

We propose **Disentangled Concept-Residual Models (D-CRMs)**, which minimize mutual information between concept and residual representations to ensure true disentanglement. Our approach improves the effectiveness of concept interventions while maintaining predictive performance.

## Key Features

- Mutual information minimization between concepts and residuals using CLUB estimator
- Improved concept intervention effectiveness
- Comprehensive evaluation framework for measuring concept-residual disentanglement
- Support for multiple datasets: CIFAR-100, CelebA, CUB-200-2011, OAI, AA2

## Methods Compared

| Method | Description |
|--------|-------------|
| CBM | Concept Bottleneck Model (no residual) |
| Latent Residual | CRM with unconstrained residual (baseline) |
| Decorrelated | CRM with cross-correlation penalty |
| IterNorm | CRM with iterative normalization |
| EYE | CRM with Expert Yielded Estimation regularization |
| Adversarial | CRM with adversarial decorrelation |
| **MI CRM (Ours)** | **D-CRM with mutual information minimization** |
| CEM | Concept Embedding Model |

## Installation

```bash
git clone https://github.com/renos/concept-residual-models.git
cd concept-residual-models
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- Ray[tune] >= 2.7
- CUDA (recommended for training)

## Dataset Setup

Set environment variables:
```bash
export CONCEPT_DATA_DIR="./data/"
export CONCEPT_SAVE_DIR="./saved/"
```

### Dataset Downloads

| Dataset | Source | Directory |
|---------|--------|-----------|
| CIFAR-100 | Auto-downloaded via torchvision | `data/cifar-100-python/` |
| CUB-200-2011 | [Caltech-UCSD Birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) | `data/CUB_200_2011/` |
| CelebA | [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | `data/celeba/` |
| OAI | [OAI (requires access)](https://nda.nih.gov/oai) | `data/OAI/` |
| AA2 | [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/) | `data/AA2/` |

## Training

### Quick Start

```bash
# Train D-CRM (MI minimization) on CIFAR-100
python train.py --config configs.cifar_baselines

# Train on CelebA
python train.py --config configs.celeba_baselines
```

### All Experiments

<details>
<summary><b>CIFAR-100</b></summary>

```bash
python train.py --config configs.cifar_baselines        # Latent Residual, Decorrelated, MI
python train.py --config configs.cifar_adversarial_decorr
python train.py --config configs.cifar_eye
python train.py --config configs.cifar_iter_norm
python train.py --config configs.cifar_cem
```
</details>

<details>
<summary><b>CelebA</b></summary>

```bash
python train.py --config configs.celeba_baselines
python train.py --config configs.celeba_adversarial_decorr
python train.py --config configs.celeba_cem
python train.py --config configs.celeba_bottleneck
```
</details>

<details>
<summary><b>CUB-200-2011</b></summary>

```bash
python train.py --config configs.cub_baselines
python train.py --config configs.cub_adversarial_decorr
python train.py --config configs.cub_eye
python train.py --config configs.cub_mi_info_bottleneck
```
</details>

<details>
<summary><b>OAI</b></summary>

```bash
python train.py --config configs.oai_baselines
python train.py --config configs.oai_adversarial_decorr
python train.py --config configs.oai_eye
python train.py --config configs.oai_iter_norm
python train.py --config configs.oai_cem
python train.py --config configs.oai_bottleneck
```
</details>

<details>
<summary><b>AA2</b></summary>

```bash
python train.py --config configs.aa2_baselines
python train.py --config configs.aa2_adversarial_decorr
python train.py --config configs.aa2_mi
python train.py --config configs.aa2_cem
python train.py --config configs.aa2_bottleneck
```
</details>

## Evaluation

Evaluate trained models:

```bash
python evaluate.py --exp-dir <EXPERIMENT_DIR> --mode <MODES>
```

**Evaluation Modes:**

| Mode | Description |
|------|-------------|
| `accuracy` | Baseline test accuracy |
| `pos_intervention` | Accuracy with ground truth concept interventions |
| `random_concepts` | Accuracy degradation with randomized concepts |
| `random_residual` | Accuracy degradation with randomized residuals |
| `mutual_info` | MI between concepts and residuals |
| `concept_pred` | F1 score for predicting concepts from residuals |
| `deeplift_shapley` | SHAP-based concept importance |

**Example:**
```bash
python evaluate.py --exp-dir ./saved/cifar_baselines/2025-01-01_12_00_00 \
    --mode accuracy pos_intervention concept_pred mutual_info
```

## Reproducing Paper Results

### Table 1: Concept-Residual Overlap Metrics
```bash
python evaluate.py --exp-dir <DIR> --mode concept_pred mutual_info --all
```

### Figure 3: Concept Prediction F1 vs SHAP Importance
```bash
python evaluate.py --exp-dir <CUB_DIR> --mode concept_pred deeplift_shapley
python scripts/plot_concept_vs_shap.py --exp-dir <CUB_DIR>
```

### Figure 4-5: Intervention Degradation Curves
```bash
python evaluate.py --exp-dir <DIR> --mode random_concepts random_residual --all
python scripts/plot_interventions.py --exp-dir <DIR>
```

### Figure 6: Concept Importance Analysis
```bash
python evaluate.py --exp-dir <DIR> --mode deeplift_shapley --all
python scripts/plot_concept_importance.py --exp-dir <DIR>
```

### Figure 7: Counterfactual Accuracy (AA2)
```bash
python evaluate.py --exp-dir <AA2_DIR> --mode test_counterfactual_2 --all
python scripts/plot_counterfactual.py --exp-dir <AA2_DIR>
```

## Project Structure

```
.
├── train.py                 # Training script with Ray Tune
├── evaluate.py              # Comprehensive evaluation suite
├── plot.py                  # Result visualization
├── lightning_ray.py         # Ray Tune + PyTorch Lightning integration
├── utils.py                 # Utility functions
├── nn_extensions.py         # Custom neural network modules
├── models/                  # Model implementations
│   ├── base.py              # ConceptModel, ConceptLightningModel
│   ├── concept_embedding.py # Concept Embedding Model (CEM)
│   ├── mutual_info.py       # MI minimization (D-CRM)
│   ├── adversarial_decorrelation.py
│   └── whitening.py         # Concept whitening
├── lib/                     # Core algorithms
│   ├── club.py              # CLUB MI estimator
│   ├── mine.py              # MINE MI estimator
│   └── iterative_normalization.py
├── datasets/                # Dataset loaders
├── experiments/             # Experiment configurations
├── configs/                 # Paper experiment configs
└── scripts/                 # Plotting scripts for paper figures
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{zabounidis2026dcrm,
  title={Disentangled Concept-Residual Models: Bridging the Interpretability--Performance Gap for Incomplete Concept Sets},
  author={Zabounidis, Renos and Oguntola, Ini and Zhao, Konghao and Campbell, Joseph and Kim, Woojun and Stepputtis, Simon and Sycara, Katia},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=NKgNizwDa6}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon concepts from:
- Concept Bottleneck Models (Koh et al., 2020)
- Concept Embedding Models (Zarlenga et al., 2022)
- CLUB MI Estimation (Cheng et al., 2020)
