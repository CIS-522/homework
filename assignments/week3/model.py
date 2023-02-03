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
        activation: Callable | list[Callable] = torch.nn.ReLU,
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

        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        if isinstance(activation, list):
            assert len(activation) == len(
                hidden_units
            ), "Number of activation functions must match number of hidden layers"

            self.activation = [a() for a in activation]
        else:
            self.activation = [activation() for _ in hidden_units]

        # add a dummy activation function for the last layer
        self.activation.append(torch.nn.Identity())

        layers_units = hidden_units + [num_classes]

        self.layers = torch.nn.ModuleList()
        current_input_size = input_size
        for layer_n_units in layers_units:
            layer = torch.nn.Linear(current_input_size, layer_n_units)

            if initializer is not None:
                initializer(layer.weight)

            self.layers.append(layer)
            current_input_size = layer_n_units

        # sanity check
        assert len(self.activation) == len(
            self.layers
        ), "Number of layers and activations must match"

    def forward(self, x: torch.Tensor) -> None:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # make sure the data has the right shape
        x = x.view(x.shape[0], -1)

        for layer, activ in zip(self.layers, self.activation):
            x = activ(layer(x))

        return x
