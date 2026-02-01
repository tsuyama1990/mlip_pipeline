from typing import Iterator, List, Literal
import numpy as np
import ase
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder

class StrainStrategy:
    """
    Strategy for generating strained structures (elasticity sampling).
    See SPEC.md Section 3.3.
    """

    def apply(
        self,
        structure: Structure,
        strain_type: Literal["uniaxial", "shear", "rattle", "volumetric"] = "uniaxial",
        magnitude: float = 0.05,
        n_points: int = 5
    ) -> Iterator[Structure]:
        """
        Apply strain strategy to the input structure.

        Args:
            structure: Base structure.
            strain_type: Type of deformation.
            magnitude: Max strain amplitude (e.g. 0.05 for 5%).
            n_points: Number of structures to generate (interpolating from -mag to +mag or 0 to mag).

        Returns:
            Iterator yielding strained Structures.
        """
        atoms = structure.to_ase()

        # Generate range of strains
        strains: np.ndarray | List[float]
        if n_points == 1:
            strains = [magnitude]
        else:
            strains = np.linspace(-magnitude, magnitude, n_points)

        if strain_type == "rattle":
            builder = StructureBuilder()
            for _ in range(n_points):
                # Magnitude interpreted as stdev for rattle
                s = builder.apply_rattle(structure, stdev=magnitude)
                yield s
            return

        original_cell = atoms.get_cell() # type: ignore[no-untyped-call]

        for eps in strains:
            if abs(eps) < 1e-6:
                continue # Skip effectively zero strain

            new_atoms = atoms.copy() # type: ignore[no-untyped-call]
            cell = original_cell.copy()
            deformation = np.eye(3)

            if strain_type == "uniaxial":
                deformation[0, 0] += eps

            elif strain_type == "volumetric":
                deformation[0, 0] += eps
                deformation[1, 1] += eps
                deformation[2, 2] += eps

            elif strain_type == "shear":
                deformation[0, 1] += eps

            new_cell = np.dot(cell, deformation)

            new_atoms.set_cell(new_cell, scale_atoms=True) # type: ignore[no-untyped-call]
            yield Structure.from_ase(new_atoms)
