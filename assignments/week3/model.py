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
        activations: Callable | list[Callable] = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.xavier_uniform_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_units: A list with the number neurons each layer. The number of layers is inferred from
                the length of the list.
            num_classes: The number of classes C.
            activations: The activation functions to use in the hidden layer. Can be a single function (it will be the
                same across all layers) or a list of functions (in this case the number of functions must match the
                number of hidden layers (hidden_units).
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        if isinstance(activations, list):
            assert len(activations) == len(
                hidden_units
            ), "Number of activation functions must match number of hidden layers"

            self.activations = [a() for a in activations]
        else:
            self.activations = [activations() for _ in hidden_units]

        # add a dummy activation function (identity) for the last layer
        self.activations.append(torch.nn.Identity())

        # list of the total number of units in each layer
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
        assert len(self.activations) == len(
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

        for layer, activ in zip(self.layers, self.activations):
            x = activ(layer(x))

        return x
