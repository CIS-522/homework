from typing import Callable

import torch


class MLP(torch.nn.Module):
    """
    An implementation of a multi-layer perceptron (MLP) for CIS-522, Homework week 3.
    """

    def __init__(
        self,
        input_size: int,
        hidden_units: int | list[int],
        num_classes: int,
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

        ## TODO: use initializer

        self.activation = activation()

        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        layers_units = [input_size] + hidden_units + [num_classes]

        self.layers = torch.nn.ModuleList()
        current_input_size = input_size
        for layer_n_units in layers_units:
            self.layers.append(torch.nn.Linear(current_input_size, layer_n_units))
            current_input_size = layer_n_units

    def forward(self, x: torch.Tensor) -> None:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            # TODO: different activations per layer (or per neuron?)
            x = self.activation(layer(x))

        return x
