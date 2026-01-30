import logging
import ase.build
import numpy as np
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger("mlip_autopipec")


class StructureBuilder:
    """
    Builder for generating and modifying atomic structures.
    Wraps ASE structure generation tools with strictly typed inputs/outputs.
    """

    def build_bulk(
        self, element: str, crystal_structure: str, lattice_constant: float
    ) -> Structure:
        """
        Build a bulk crystal structure.

        Args:
            element: Chemical symbol (e.g., "Si")
            crystal_structure: Crystal structure (e.g., "diamond", "fcc")
            lattice_constant: Lattice constant in Angstroms

        Returns:
            Structure: The generated structure.
        """
        atoms = ase.build.bulk(
            name=element,
            crystalstructure=crystal_structure,
            a=lattice_constant,
            cubic=True,  # Force cubic cell for simplicity in Cycle 02
        )
        return Structure.from_ase(atoms)

    def apply_rattle(
        self, structure: Structure, stdev: float, seed: int = 42
    ) -> Structure:
        """
        Apply random thermal noise (rattle) to atomic positions.

        Args:
            structure: Input structure
            stdev: Standard deviation of the noise (Angstroms)
            seed: Random seed for reproducibility

        Returns:
            Structure: A new structure with rattled positions.
        """
        atoms = structure.to_ase()

        # ASE rattle takes an integer seed or RandomState.
        # Memory says "strictly requires an integer seed".
        atoms.rattle(stdev=stdev, seed=seed)  # type: ignore[no-untyped-call]

        self._validate_structure(atoms)

        return Structure.from_ase(atoms)

    def _validate_structure(self, atoms: ase.Atoms, min_dist: float = 0.5) -> None:
        """
        Validate that the structure is physically reasonable.

        Args:
            atoms: ASE Atoms object
            min_dist: Minimum allowed distance between atoms in Angstroms.

        Raises:
            ValueError: If atoms are too close.
        """
        # Use mic=True to respect Periodic Boundary Conditions
        dist_matrix = atoms.get_all_distances(mic=True)  # type: ignore[no-untyped-call]

        # Mask diagonal (distance to self is 0)
        np.fill_diagonal(dist_matrix, np.inf)

        min_d = np.min(dist_matrix)
        if min_d < min_dist:
            logger.warning(
                f"Unphysical atomic distances detected: {min_d:.3f} A < {min_dist} A"
            )
            # For Cycle 02, we warn. In active learning, we might reject.
            # But strictly speaking, if it's too close, LAMMPS will crash or explode.
            # Let's raise an error to enforce "Physically Reasonable" as per User request.
            raise ValueError(f"Atoms are too close: {min_d:.3f} A < {min_dist} A")
