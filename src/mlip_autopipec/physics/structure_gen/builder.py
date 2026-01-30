import ase.build
import numpy as np

from mlip_autopipec.domain_models.structure import Structure


class StructureBuilder:
    """
    Builder for generating atomic structures.
    Wraps ASE's build functions and ensures strict Type adherence.
    """

    def build_bulk(
        self,
        element: str,
        crystal_structure: str,
        lattice_constant: float,
        cubic: bool = False
    ) -> Structure:
        """
        Build a bulk crystal structure.

        Args:
            element: Chemical symbol (e.g. "Si").
            crystal_structure: Crystal structure (e.g. "diamond", "fcc").
            lattice_constant: Lattice constant in Angstroms.
            cubic: If True, returns the conventional cubic cell.

        Returns:
            A Structure object.
        """
        # ase.build.bulk args: name, crystalstructure, a, cubic
        atoms = ase.build.bulk(
            name=element,
            crystalstructure=crystal_structure,
            a=lattice_constant,
            cubic=cubic
        )
        return Structure.from_ase(atoms)

    def apply_rattle(self, structure: Structure, stdev: float, seed: int) -> Structure:
        """
        Apply random thermal noise (rattle) to positions.

        Args:
            structure: Input structure.
            stdev: Standard deviation of noise (Angstroms).
            seed: Random seed for reproducibility.

        Returns:
            New Structure object with rattled positions.
        """
        atoms = structure.to_ase()
        rng = np.random.RandomState(seed)
        atoms.rattle(stdev=stdev, rng=rng) # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)

    def apply_strain(self, structure: Structure, strain_tensor: np.ndarray) -> Structure:
        """
        Apply strain to the unit cell.

        Args:
            structure: Input structure.
            strain_tensor: 3x3 strain tensor (epsilon).
                           cell_new = cell_old @ (I + epsilon)
        """
        atoms = structure.to_ase()
        cell = atoms.get_cell() # type: ignore[no-untyped-call]
        deformation = np.eye(3) + strain_tensor
        new_cell = cell @ deformation
        atoms.set_cell(new_cell, scale_atoms=True) # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)
