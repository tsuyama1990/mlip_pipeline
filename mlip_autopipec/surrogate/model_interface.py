from typing import Protocol, runtime_checkable

import numpy as np
from ase import Atoms


@runtime_checkable
class ModelInterface(Protocol):
    """Protocol for surrogate models."""

    def load_model(self, model_path: str, device: str) -> None:
        """
        Loads the model weights.

        Args:
            model_path: Path or identifier for the model.
            device: 'cpu' or 'cuda'.
        """
        ...

    def compute_energy_forces(self, atoms_list: list[Atoms]) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Computes energy and forces for a batch of structures.

        Args:
            atoms_list: List of ASE Atoms objects.

        Returns:
            Tuple of (energies, forces_list).
            energies: 1D numpy array of shape (N,) in eV.
            forces_list: List of N numpy arrays, each shape (n_atoms, 3) in eV/A.
        """
        ...

    def compute_descriptors(self, atoms_list: list[Atoms]) -> np.ndarray:
        """
        Computes descriptors for sampling.

        Args:
            atoms_list: List of ASE Atoms objects.

        Returns:
            2D numpy array of shape (N, D) where N is number of structures and D is descriptor dimension.
        """
        ...
