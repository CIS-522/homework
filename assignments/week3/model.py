import torch
import torch.nn.functional as F
from typing import Callable


class MLP(torch.nn.Module):
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
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, num_classes)
        self.activation = activation
        self.initializer = initializer
        self.hidden_count = hidden_count
        # dropout prevents overfitting of data
        self.droput = torch.nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # x = self.initializer(x)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        for i in range(self.hidden_count):
            x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
