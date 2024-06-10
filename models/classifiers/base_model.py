from typing import Dict

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from models.model import HaGRIDModel


from typing import Dict

import torch
from torch import Tensor, nn

class ClassifierModel(nn.Module):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__()
        self.hagrid_model = model(**kwargs)

    def forward(self, image_tensor: Tensor, targets: Dict = None) -> Dict:
        """
        Forward pass of the classifier model.

        Parameters
        ----------
        image_tensor: Tensor
            A tensor representing a batch of images.
        targets: Dict, optional
            Dictionary containing the labels for the images.

        Returns
        -------
        Dict
            A dictionary containing the model's output, and optionally loss if targets are provided.
        """
        model_output = self.hagrid_model(image_tensor)
        result = {"labels": model_output}

        if targets is not None:
            target_tensor = torch.stack([target["labels"] for target in targets])
            loss = self.criterion(result["labels"], target_tensor)
            result['loss'] = loss

        return result

    def criterion(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """
        Calculate loss for the given outputs and targets.
        
        Parameters
        ----------
        outputs: Tensor
            The model's logits.
        targets: Tensor
            The targets associated with the inputs.

        Returns
        -------
        Tensor
            The loss value as a tensor.
        """
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets)
