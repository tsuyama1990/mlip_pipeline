import logging
from collections.abc import Iterator

import numpy as np

from mlip_autopipec.domain_models.config import ActiveLearningConfig
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)

class CandidateGenerator:
    """
    Generates local candidates around a seed structure for Active Learning.
    """

    def __init__(self, config: ActiveLearningConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng()

    def generate_local(self, structure: Structure) -> Iterator[Structure]:
        """
        Generates local candidates using the configured method (default: perturbation).

        Args:
            structure: The seed structure (e.g. from a halt).

        Returns:
            An iterator of perturbed Structure objects.
        """
        method = self.config.sampling_method
        if method == "perturbation":
            yield from self._generate_perturbations(structure)
        else:
            logger.warning(f"Unknown sampling method '{method}', falling back to perturbation.")
            yield from self._generate_perturbations(structure)

    def _generate_perturbations(self, structure: Structure) -> Iterator[Structure]:
        """
        Generates candidates by random atomic displacement.

        This method creates copies of the seed structure and applies Gaussian noise
        to the atomic positions. The magnitude of the noise is controlled by
        `config.perturbation_magnitude`.

        Args:
            structure: The seed structure to perturb.

        Yields:
            Perturbed Structure objects.
        """
        count = self.config.n_candidates
        magnitude = self.config.perturbation_magnitude

        # Get ASE atoms
        original_atoms = structure.to_ase()

        # Ensure we have positions
        if original_atoms.positions is None or len(original_atoms) == 0:
            logger.warning("Structure has no atoms. Returning seed only.")
            yield structure
            return

        for i in range(count):
            # Create copy using ASE copy (explicit type ignore handled in Structure.to_ase but here we use ase directly)
            atoms = original_atoms.copy() # type: ignore[no-untyped-call]

            # Apply random displacement
            # Use standard normal * magnitude
            displacement = self._rng.normal(0, magnitude, atoms.positions.shape)
            atoms.positions += displacement

            yield Structure(
                atoms=atoms,
                provenance=f"{structure.provenance}_local_{i}",
                label_status="unlabeled",
                metadata={"source_seed_provenance": structure.provenance}
            )
