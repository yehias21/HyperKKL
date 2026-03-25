"""Normalizer module for standardizing input/output tensors."""

import torch
import torch.nn as nn
from torch import Tensor


class Normalizer(nn.Module):
    """Normalizes and denormalizes tensors using dataset statistics.

    Stores per-variable mean and standard deviation for both *normal* and
    *physics* modes, for x-data and z-data.  All statistics are kept as
    registered buffers so they travel with the model across devices and
    are included in ``state_dict``.

    Parameters
    ----------
    dataset : object
        A dataset that exposes the following attributes:

        * ``system.x_size`` / ``system.z_size`` -- dimensionality of x / z.
        * ``mean_x``, ``std_x``, ``mean_z``, ``std_z`` -- normal-mode stats.
        * ``mean_x_ph``, ``std_x_ph``, ``mean_z_ph``, ``std_z_ph`` --
          physics-mode stats.
    """

    def __init__(self, dataset: object) -> None:
        super().__init__()

        self.x_size: int = dataset.system.x_size
        self.z_size: int = dataset.system.z_size

        # Normal-mode statistics
        self.register_buffer("mean_x", dataset.mean_x)
        self.register_buffer("std_x", dataset.std_x)
        self.register_buffer("mean_z", dataset.mean_z)
        self.register_buffer("std_z", dataset.std_z)

        # Physics-mode statistics
        self.register_buffer("mean_x_ph", dataset.mean_x_ph)
        self.register_buffer("std_x_ph", dataset.std_x_ph)
        self.register_buffer("mean_z_ph", dataset.mean_z_ph)
        self.register_buffer("std_z_ph", dataset.std_z_ph)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def check_sys(self, tensor: Tensor, mode: str) -> tuple[Tensor, Tensor]:
        """Return the appropriate (mean, std) pair for *tensor* and *mode*.

        The feature dimension (``tensor.size(1)``) is compared against
        ``x_size`` and ``z_size`` to decide whether the tensor represents
        x-data or z-data.  *mode* selects between ``'normal'`` and
        ``'physics'`` statistics.

        Parameters
        ----------
        tensor : Tensor
            Input tensor of shape ``(batch, features)``.
        mode : str
            ``'normal'`` or ``'physics'``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(mean, std)`` tensors broadcastable to *tensor*.

        Raises
        ------
        ValueError
            If the feature dimension does not match either ``x_size`` or
            ``z_size``.
        """
        feat_dim = tensor.size(1)

        if feat_dim == self.x_size:
            if mode == "physics":
                return self.mean_x_ph, self.std_x
            return self.mean_x, self.std_x

        if feat_dim == self.z_size:
            if mode == "physics":
                return self.mean_z_ph, self.std_z_ph
            return self.mean_z, self.std_z

        raise ValueError(
            f"Feature dimension {feat_dim} does not match x_size "
            f"({self.x_size}) or z_size ({self.z_size})."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Normalize(self, tensor: Tensor, mode: str = "normal") -> Tensor:
        """Standardize *tensor* to zero mean and unit variance.

        Parameters
        ----------
        tensor : Tensor
            Raw tensor of shape ``(batch, features)``.
        mode : str
            ``'normal'`` or ``'physics'``.

        Returns
        -------
        Tensor
            Normalized tensor.
        """
        mean, std = self.check_sys(tensor, mode)
        return (tensor - mean) / std

    def Denormalize(self, tensor: Tensor, mode: str = "normal") -> Tensor:
        """Invert the standardization applied by :meth:`Normalize`.

        Parameters
        ----------
        tensor : Tensor
            Normalized tensor of shape ``(batch, features)``.
        mode : str
            ``'normal'`` or ``'physics'``.

        Returns
        -------
        Tensor
            Tensor in original (un-normalized) scale.
        """
        mean, std = self.check_sys(tensor, mode)
        return tensor * std + mean
