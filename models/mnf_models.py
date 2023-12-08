from typing import Any

from torch import nn

from models.torch_mnf.layers import MNFLinear


class MNFNet(nn.Sequential):
    """Bayesian DNN with parameter posteriors modeled by normalizing flows."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model."""
        layers = [
            MNFLinear(15, 64,**kwargs),
            nn.SELU(),
            MNFLinear(64, 128, **kwargs),
            nn.SELU(),
            MNFLinear(128, 256, **kwargs),
            nn.SELU(),
            MNFLinear(256, 128, **kwargs),
            nn.SELU(),
            MNFLinear(128, 64, **kwargs),
            nn.SELU(),
            MNFLinear(64, 3, **kwargs)
        ]
        super().__init__(*layers)

    def kl_div(self) -> float:
        """Compute current KL divergence of the whole model. Given by the sum
        of KL divs. from each MNF layer. Use as a regularization term during training.
        """
        return sum(lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div"))
