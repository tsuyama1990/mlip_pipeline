import ase.build
import numpy as np
from mlip_autopipec.domain_models.structure import Structure


class StructureBuilder:
    def build_bulk(self, element: str, crystal_structure: str, lattice_constant: float, cubic: bool = False) -> Structure:
        """
        Build a bulk structure using ASE.
        """
        atoms = ase.build.bulk(
            name=element,
            crystalstructure=crystal_structure,
            a=lattice_constant,
            cubic=cubic
        )
        return Structure.from_ase(atoms)

    def apply_rattle(self, structure: Structure, stdev: float, seed: int = 42) -> Structure:
        """
        Apply random displacement to atoms.
        """
        atoms = structure.to_ase()
        atoms.rattle(stdev=stdev, seed=seed)  # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)

    def apply_strain(self, structure: Structure, strain_tensor: np.ndarray) -> Structure:
        """
        Apply strain to the structure.
        strain_tensor: 3x3 strain matrix (epsilon).
        """
        atoms = structure.to_ase()
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]

        # strain_tensor is 3x3 epsilon
        deformation = np.eye(3) + strain_tensor

        # Apply deformation to cell
        # ASE cell is row-major: [v1, v2, v3]
        # v_new = (I+e) * v
        # v_new (row) = v (row) * (I+e).T
        new_cell = cell @ deformation.T

        atoms.set_cell(new_cell, scale_atoms=True)  # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)
