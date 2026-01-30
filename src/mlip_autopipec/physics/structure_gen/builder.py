from typing import Optional

import numpy as np
from ase.build import bulk

from mlip_autopipec.domain_models.structure import Structure


class StructureBuilder:
    """
    Builder for generating and modifying atomic structures.
    Wraps ASE functionalities with strict type safety and reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def build_bulk(self, element: str, crystal_structure: str, lattice_constant: float) -> Structure:
        """
        Build a perfect bulk crystal structure.

        Args:
            element: Chemical symbol (e.g., "Si").
            crystal_structure: Crystal structure (e.g., "diamond", "fcc").
            lattice_constant: Lattice constant in Angstroms.

        Returns:
            A new Structure object.
        """
        atoms = bulk(element, crystalstructure=crystal_structure, a=lattice_constant)
        return Structure.from_ase(atoms)

    def apply_rattle(self, structure: Structure, stdev: float) -> Structure:
        """
        Apply random thermal noise (rattle) to atomic positions.

        Args:
            structure: The input structure.
            stdev: Standard deviation of the noise in Angstroms.

        Returns:
            A new Structure object with modified positions.
        """
        atoms = structure.to_ase()
        # ASE rattle uses numpy random. We can seed it by setting the global seed
        # OR passing a seed. ASE rattle doc says "rng: Random number generator".
        # Let's see if we can pass self.rng.

        # In newer ASE, rattle accepts `rng`. In older, might be `seed`.
        # We will use `rng=self.rng` if supported, else `seed`.
        # But ASE `atoms.rattle` signature: (stdev=0.001, seed=None, rng=None)

        atoms.rattle(stdev=stdev, rng=self.rng)  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)

    def apply_strain(self, structure: Structure, strain_tensor: np.ndarray) -> Structure:
        """
        Apply a strain tensor to the simulation cell.

        Args:
            structure: The input structure.
            strain_tensor: 3x3 deformation gradient tensor (F) or strain tensor (epsilon).
                           Here we assume F (Deformation Gradient), so New = Old * F.
                           Or strictly strain? SPEC says "Deforms the cell".
                           Usually Cell_new = Cell_old @ strain_tensor.

        Returns:
            A new Structure object with deformed cell.
        """
        if strain_tensor.shape != (3, 3):
            raise ValueError("Strain tensor must be 3x3")

        atoms = structure.to_ase()
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]

        # Apply deformation: cell_new = cell_old @ strain_tensor
        # Note: ASE set_cell(scale_atoms=True) scales positions too.
        new_cell = cell @ strain_tensor

        atoms.set_cell(new_cell, scale_atoms=True)  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)
