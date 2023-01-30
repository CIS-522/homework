"""
Run the MLP training and evaluation pipeline.
"""

from model_factory import create_model

# MNIST:
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

# PyTorch:
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Other:
from typing import Tuple
from tqdm import tqdm

# The transform list is a set of operations that we apply to the data
# before we use it. In this case, we convert the data to a tensor and
# flatten it. (Thought-exercise: Why do we need to flatten the data?)
_transform_list = [
    ToTensor(),
    lambda x: x.view(-1),
]


def get_mnist_data() -> Tuple[DataLoader, DataLoader]:
    """
    Get the MNIST data from torchvision.

    Arguments:
        None

    Returns:
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.

    """
    # Get the training data:
    train_data = MNIST(
        root="data", train=True, download=True, transform=Compose(_transform_list)
    )
    # Create a data loader for the training data:
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # Get the test data:
    test_data = MNIST(
        root="data", train=False, download=True, transform=Compose(_transform_list)
    )
    # Create a data loader for the test data:
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    # Return the data loaders:
    return train_loader, test_loader


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    """
    Train a model on the MNIST data.

    Arguments:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        device (torch.device): The device to use for training.

    Returns:
        None

    """
    # Create an optimizer:
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # Create a loss function:
    criterion = CrossEntropyLoss()
    # Move the model to the device:
    model.to(device)
    # Create a progress bar:
    progress_bar = tqdm(range(num_epochs))
    # Train the model:
    for epoch in progress_bar:
        # Set the model to training mode:
        model.train()
        # Iterate over the training data:
        for batch in train_loader:
            # Get the data and labels:
            data, labels = batch
            # Move the data and labels to the device:
            data = data.to(device)
            labels = labels.to(device)
            # Zero the gradients:
            optimizer.zero_grad()
            # Forward pass:
            outputs = model(data)
            # Calculate the loss:
            loss = criterion(outputs, labels)
            # Backward pass:
            loss.backward()
            # Update the parameters:
            optimizer.step()
        # Set the model to evaluation mode:
        model.eval()

        # Calculate the accuracy on the test data:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                # Get the data and labels:
                data, labels = batch
                # Move the data and labels to the device:
                data = data.to(device)
                labels = labels.to(device)
                # Forward pass:
                outputs = model(data)
                # Get the predictions:
                _, predictions = torch.max(outputs.data, 1)
                # Update the total and correct counts:
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        # Calculate the accuracy:
        accuracy = correct / total
        # Update the progress bar:
        progress_bar.set_description(f"Epoch: {epoch}, Accuracy: {accuracy:.4f}")


def main():
    # Get the data:
    train_loader, test_loader = get_mnist_data()
    # Create the model:
    model = create_model(784, 10)
    # Train the model:
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,
        learning_rate=0.001,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    main()
