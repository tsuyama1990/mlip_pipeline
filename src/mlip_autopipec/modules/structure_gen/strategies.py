"""Structure generation strategies."""

import logging

import numpy as np
from ase.build import bulk

from mlip_autopipec.domain_models.config import ExplorationConfig
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class ColdStartStrategy:
    """Strategy for generating initial structures from scratch."""

    def generate(self, config: ExplorationConfig) -> list[Structure]:
        """Generate initial structures based on composition."""
        if not config.composition:
            logger.warning("No composition provided for Cold Start. Returning empty.")
            return []

        structures = []
        try:
            # Generate 'num_candidates' variants by supercell/rattle.
            # Using ase.build.bulk handles crystal structures.
            base_atoms = bulk(config.composition)

            for _ in range(config.num_candidates):
                atoms = base_atoms.copy()  # type: ignore[no-untyped-call]

                # Supercell
                if config.supercell_size != [1, 1, 1]:
                    atoms = atoms.repeat(config.supercell_size)  # type: ignore[no-untyped-call]

                # Rattle
                if config.rattle_amplitude > 0:
                    atoms.rattle(stdev=config.rattle_amplitude)  # type: ignore[no-untyped-call]

                structures.append(Structure.from_ase(atoms))

        except Exception as e:
            logger.error(f"Failed to generate structure for {config.composition}: {e}")
            # Return what we have or empty
            return structures

        return structures


class RandomPerturbationStrategy:
    """Strategy for perturbing existing structures."""

    def apply(self, structure: Structure, config: ExplorationConfig) -> Structure:
        """Apply random perturbation (rattle + strain)."""
        atoms = structure.to_ase()

        # Rattle
        if config.rattle_amplitude > 0:
            atoms.rattle(stdev=config.rattle_amplitude)  # type: ignore[no-untyped-call]

        # Strain (Cell deformation)
        # Apply a random strain tensor ~ N(0, 0.02)
        strain_tensor = np.eye(3) + np.random.normal(0, 0.02, (3, 3))  # 2% strain

        # Ensure volume doesn't collapse (determinant > 0)
        if np.linalg.det(strain_tensor) < 0.1:
             strain_tensor = np.eye(3) # Fallback

        original_cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        new_cell = np.dot(original_cell, strain_tensor)

        atoms.set_cell(new_cell, scale_atoms=True)  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)
