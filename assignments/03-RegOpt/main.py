"""
A complete implementation and training of a CIFAR10 classifier.

The prompt is to create another LearningRateScheduler.

"""
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from model import MiniCNN
from scheduler import CustomLRScheduler
from config import CONFIG

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar10_data() -> Tuple[DataLoader, DataLoader]:
    """
    Get the CIFAR10 data from torchvision.

    Arguments:
        None

    Returns:
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.

    """
    # Get the training data:
    train_data = CIFAR10(
        root="data/cifar10", train=True, download=True, transform=CONFIG.transforms
    )
    # Create a data loader for the training data:
    train_loader = DataLoader(train_data, batch_size=CONFIG.batch_size, shuffle=True)
    # Get the test data:
    test_data = CIFAR10(
        root="data/cifar10", train=False, download=True, transform=CONFIG.transforms
    )
    # Create a data loader for the test data:
    test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size, shuffle=True)
    # Return the data loaders:
    return train_loader, test_loader


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device = device,
) -> None:
    """
    Train a model on the data.

    Arguments:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.
        num_epochs (int): The number of epochs to train for.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (torch.nn.Module): The loss function to use.
        learning_rate_scheduler (torch.optim.lr_scheduler._LRScheduler): The
            learning rate scheduler to use.
        device (torch.device): The device to use for training.

    Returns:
        None

    """
    # Move the model to the device:
    model.to(device)
    # Loop over the epochs:
    for epoch in range(num_epochs):
        # Set the model to training mode:
        model.train()
        # Loop over the training data:
        for x, y in tqdm(train_loader):
            # Move the data to the device:
            x, y = x.to(device), y.to(device)
            # Zero the gradients:
            optimizer.zero_grad()
            # Forward pass:
            y_hat = model(x)
            # Compute the loss:
            loss = criterion(y_hat, y)
            # Backward pass:
            loss.backward()
            # Update the parameters:
            optimizer.step()
            # Update the learning rate:
            learning_rate_scheduler.step()
        # Set the model to evaluation mode:
        model.eval()
        # Compute the accuracy on the test data:
        accuracy = compute_accuracy(model, test_loader, device)
        # Print the results:
        print(f"Epoch {epoch + 1} | Test Accuracy: {accuracy:.2f}")


def compute_accuracy(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device = device
) -> float:
    """
    Compute the accuracy of a model on some data.

    Arguments:
        model (torch.nn.Module): The model to compute the accuracy of.
        data_loader (DataLoader): The data loader to use.
        device (torch.device): The device to use for training.

    Returns:
        accuracy (float): The accuracy of the model on the data.

    """
    # Set the model to evaluation mode:
    model.eval()
    # Initialize the number of correct predictions:
    num_correct = 0
    # Loop over the data:
    for x, y in data_loader:
        # Move the data to the device:
        x, y = x.to(device), y.to(device)
        # Forward pass:
        y_hat = model(x)
        # Compute the predictions:
        predictions = torch.argmax(y_hat, dim=1)
        # Update the number of correct predictions:
        num_correct += torch.sum(predictions == y).item()
    # Compute the accuracy:
    accuracy = num_correct / len(data_loader.dataset)
    # Return the accuracy
    return accuracy


def main() -> None:
    """
    Train a model on the data.

    Arguments:
        None

    Returns:
        None

    """
    # Create the data loaders:
    train_loader, test_loader = get_cifar10_data()
    # Create the model:
    model = MiniCNN(num_channels=3)
    # Create the optimizer:
    optimizer = CONFIG.optimizer_factory(model)
    # Create the loss function:
    criterion = torch.nn.CrossEntropyLoss()
    # Create the learning rate scheduler:
    learning_rate_scheduler = CustomLRScheduler(optimizer, **CONFIG.lrs_kwargs)
    # Train the model:
    train(
        model,
        train_loader,
        test_loader,
        num_epochs=CONFIG.num_epochs,
        optimizer=optimizer,
        criterion=criterion,
        learning_rate_scheduler=learning_rate_scheduler,
    )


if __name__ == "__main__":
    main()
