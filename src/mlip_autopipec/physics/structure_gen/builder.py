import numpy as np
from ase.build import bulk
from mlip_autopipec.domain_models.structure import Structure


class StructureBuilder:
    """Builder for generating atomic structures."""

    def build_bulk(self, element: str, crystal_structure: str, lattice_constant: float) -> Structure:
        """
        Build a bulk crystal structure.

        Args:
            element: Chemical symbol (e.g., 'Si').
            crystal_structure: Crystal structure (e.g., 'diamond', 'fcc').
            lattice_constant: Lattice constant in Angstroms.

        Returns:
            Structure: The generated structure.
        """
        # ase.build.bulk args: name, crystalstructure, a, ...
        # forcing cubic=True for simplicity as per common use cases in this pipeline
        atoms = bulk(element, crystalstructure=crystal_structure, a=lattice_constant, cubic=True) # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)

    def apply_rattle(self, structure: Structure, stdev: float, seed: int = 42) -> Structure:
        """
        Apply random displacement to atoms.

        Args:
            structure: Input structure.
            stdev: Standard deviation of the displacement.
            seed: Random seed.

        Returns:
            Structure: Rattled structure.
        """
        atoms = structure.to_ase()
        # ASE's rattle takes stdev and seed.
        atoms.rattle(stdev=stdev, seed=seed) # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)

    def apply_strain(self, structure: Structure, strain_tensor: np.ndarray) -> Structure:
        """
        Apply strain to the cell.

        Args:
            structure: Input structure.
            strain_tensor: 3x3 strain matrix (epsilon).

        Returns:
            Structure: Strained structure.
        """
        # deformation = I + epsilon
        deformation = np.eye(3) + strain_tensor
        atoms = structure.to_ase()
        cell = atoms.get_cell() # type: ignore[no-untyped-call]
        new_cell = np.dot(cell, deformation)
        atoms.set_cell(new_cell, scale_atoms=True) # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)
