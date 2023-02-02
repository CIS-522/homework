import torch
from typing import Callable

import torch


class MLP(torch.nn.Module):
    """
    An implementation of a multi-layer perceptron (MLP) for CIS-522, Homework week 3.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

    def forward(self, x: torch.Tensor) -> None:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        ...
