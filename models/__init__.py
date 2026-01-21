"""
Model components for concept residual architectures.

This package provides:
- ConceptModel: Base concept bottleneck model
- ConceptEmbeddingModel: Concept Embedding Model (CEM)
- MutualInfoConceptLightningModel: D-CRM with MI minimization
- AdversarialDecorrelationConceptLightningModel: Adversarial residual disentanglement
- ConceptWhiteningLightningModel: Concept whitening model
"""

from .base import ConceptBatch, ConceptModel, ConceptLightningModel
from .bottleneck import make_bottleneck_layer
from .mutual_info import MutualInfoConceptLightningModel
from .adversarial_decorrelation import AdversarialDecorrelationConceptLightningModel
from .whitening import ConceptWhiteningLightningModel
from .concept_embedding import ConceptEmbeddingModel