from abc import ABC, abstractmethod

import ase.build
import numpy as np

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.config import StructureGenConfig


class GenerationStrategy(ABC):
    """Abstract base class for structure generation strategies."""

    @abstractmethod
    def generate(self, config: StructureGenConfig) -> Structure:
        """Generate a structure based on configuration."""
        pass


class ColdStartStrategy(GenerationStrategy):
    """Generates initial structures using ASE bulk."""

    def generate(self, config: StructureGenConfig) -> Structure:
        """
        Generate a bulk structure supercell.
        """
        atoms = ase.build.bulk(
            config.element,
            crystalstructure=config.crystal_structure,
            a=config.lattice_constant,
            cubic=True
        )  # type: ignore[no-untyped-call]

        if config.supercell != (1, 1, 1):
            # atoms * supercell returns a new Atoms object
            atoms = atoms * config.supercell  # type: ignore[operator]

        return Structure.from_ase(atoms)


class RattleStrategy:
    """Applies random displacement to atoms."""

    def apply(self, structure: Structure, stdev: float, seed: int = 42) -> Structure:
        """
        Apply thermal noise (rattle) to the structure.
        """
        atoms = structure.to_ase()
        rng = np.random.default_rng(seed)
        atoms.rattle(stdev=stdev, rng=rng)  # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)
