"""Feedforward neural network with optional normalizer integration."""

from typing import Optional

import torch
from torch import nn


class NN(nn.Module):
    """Feedforward neural network with configurable depth, width, and activation.

    Supports an optional normalizer object that normalizes inputs before the
    forward pass and denormalizes outputs afterwards.  The ``mode`` attribute
    (``'normal'`` or ``'physics'``) is forwarded to the normalizer so it can
    select the appropriate statistics.

    Parameters
    ----------
    num_hidden : int
        Number of hidden layers.
    hidden_size : int
        Number of units in each hidden layer.
    in_size : int
        Dimensionality of the input.
    out_size : int
        Dimensionality of the output.
    activation : nn.Module
        Activation function applied after each hidden layer.
    dropout_prob : float, optional
        Dropout probability (currently unused, reserved for future use).
    normalizer : object or None, optional
        An object exposing ``Normalize(tensor, mode)`` and
        ``Denormalize(tensor, mode)`` methods.  When provided, the forward
        pass will normalize inputs and denormalize outputs automatically.
    """

    def __init__(
        self,
        num_hidden: int,
        hidden_size: int,
        in_size: int,
        out_size: int,
        activation: nn.Module,
        dropout_prob: float = 0.0,
        normalizer: Optional[object] = None,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = activation
        self.normalizer = normalizer
        self.mode: str = "normal"

        current_dim = in_size
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(current_dim, hidden_size))
            current_dim = hidden_size
        self.layers.append(nn.Linear(current_dim, out_size))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        If a normalizer is attached, the input is normalized before being fed
        through the network and the output is denormalized before being
        returned.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape ``(batch, in_size)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, out_size)``.
        """
        if self.normalizer is not None:
            tensor = self.normalizer.Normalize(tensor, self.mode).float()

        tensor = self.layers[0](tensor)
        for layer in self.layers[1:-1]:
            tensor = self.activation(layer(tensor))
        tensor = self.layers[-1](tensor)

        if self.normalizer is not None:
            tensor = self.normalizer.Denormalize(tensor, self.mode).float()

        return tensor
