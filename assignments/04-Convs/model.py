import torch


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
