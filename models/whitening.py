import torch
from typing import Any

from .base import ConceptModel, ConceptLightningModel
from .bottleneck import ConceptWhitening
from utils import unwrap



class ConceptWhiteningLightningModel(ConceptLightningModel):
    """
    Concept model that uses concept whitening to decorrelate, normalize,
    and align the latent space with concepts.
    """

    def __init__(
        self,
        concept_model: ConceptModel,
        cw_alignment_frequency: int = 20,
        **kwargs):
        """
        Parameters
        ----------
        concept_model : ConceptModel
            Concept model
        cw_alignment_frequency : int
            Frequency of concept alignment (e.g. every N batches)
        """
        assert isinstance(unwrap(concept_model.bottleneck_layer), ConceptWhitening)
        super().__init__(concept_model, **dict(kwargs, concept_loss_fn=None))
        self.alignment_frequency = cw_alignment_frequency

    def on_train_start(self):
        """
        Create concept data loaders.
        """
        # Get training data loader
        loader = self.trainer.fit_loop._data_source.dataloader()
        (data, concepts), targets = next(iter(loader))
        batch_size, concept_dim = concepts.shape

        # Create concept data loaders (one for each concept)
        self.concept_loaders = []
        try:
            for concept_idx in range(concept_dim):
                concept_loader = torch.utils.data.DataLoader(
                    dataset=[x for ((x, c), y) in loader.dataset if c[concept_idx] == 1],
                    batch_size=batch_size,
                    shuffle=True,
                )
                self.concept_loaders.append(concept_loader)

        except ValueError as e:
            print('Error creating concept loaders:', e)
            self.concept_loaders = None

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """
        Align concepts in the concept whitening layer.

        Parameters
        ----------
        batch : ConceptBatch
            Batch of ((data, concepts), targets)
        batch_idx : int
            Batch index
        """
        if self.concept_loaders is None:
            return

        if (batch_idx + 1) % self.alignment_frequency == 0:
            self.freeze()
            with torch.no_grad():
                for concept_idx, concept_loader in enumerate(self.concept_loaders):
                    self.concept_model.bottleneck_layer.mode = concept_idx
                    for X in concept_loader:
                        X = X.requires_grad_().to(self.device)
                        self.concept_model(X)
                        break

                    self.concept_model.bottleneck_layer.update_rotation_matrix()
                    self.concept_model.bottleneck_layer.mode = -1

            self.unfreeze()
