from abc import ABC, abstractmethod

import numpy as np

from mlip_autopipec.domain_models.structure import Structure


class StructureTransform(ABC):
    """Abstract base class for structure transformations."""

    @abstractmethod
    def apply(self, structure: Structure) -> Structure:
        """
        Apply transformation to a structure.

        Args:
            structure: Input structure.

        Returns:
            Transformed structure (new instance).
        """
        ...


class RattleTransform(StructureTransform):
    """Applies random Gaussian noise to atomic positions."""

    def __init__(self, stdev: float) -> None:
        self.stdev = stdev

    def apply(self, structure: Structure) -> Structure:
        if self.stdev <= 0:
            return structure

        new_s = structure.model_copy(deep=True)
        noise = np.random.normal(0.0, self.stdev, structure.positions.shape)
        new_s.positions = new_s.positions + noise

        # Update tags
        new_s.tags["rattled"] = True
        new_s.tags["rattle_stdev"] = self.stdev

        return new_s


class StrainTransform(StructureTransform):
    """Applies random strain to the cell and positions."""

    def __init__(self, strain_range: float) -> None:
        self.strain_range = strain_range

    def apply(self, structure: Structure) -> Structure:
        if self.strain_range <= 0:
            return structure

        new_s = structure.model_copy(deep=True)

        # Generate random deformation gradient F = I + epsilon
        # epsilon components in [-strain_range, strain_range]
        epsilon = (np.random.rand(3, 3) - 0.5) * 2 * self.strain_range
        F = np.eye(3) + epsilon

        # Apply deformation
        # positions' = positions @ F.T (since positions are row vectors)
        # cell' = cell @ F.T
        new_s.positions = new_s.positions @ F.T
        new_s.cell = new_s.cell @ F.T

        # Update tags
        new_s.tags["strained"] = True
        new_s.tags["strain_range"] = self.strain_range

        return new_s
