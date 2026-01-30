import logging
import ase.build
import ase.neighborlist
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
        # Use NeighborList for O(N) validation instead of O(N^2) distance matrix
        # Cutoff is min_dist / 2 because NeighborList uses cutoff radii (r_i + r_j)
        # We want to find any pair with distance < min_dist.
        # So if we set cutoff to min_dist/2 for all atoms, any pair closer than min_dist will be found.
        # Wait, neighbor_list takes cutoffs dictionary or list.
        # Simpler: ase.neighborlist.neighbor_list('d', atoms, cutoff) returns distances < cutoff.

        # 'd' returns distances.
        distances = ase.neighborlist.neighbor_list('d', atoms, cutoff=min_dist)

        # Filter out self-interactions if any (neighbor_list usually doesn't include self unless specified)
        # But neighbor_list returns list of distances.
        if len(distances) > 0:
            min_d = np.min(distances)
            # neighbor_list can return 0 if atoms overlap perfectly? No, self interaction usually handled.
            # But let's check.
            # If atoms are distinct, dist > 0.

            logger.warning(
                f"Unphysical atomic distances detected: {min_d:.3f} A < {min_dist} A"
            )
            raise ValueError(f"Atoms are too close: {min_d:.3f} A < {min_dist} A")
