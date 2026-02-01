from typing import List, Literal
import numpy as np
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
    ) -> List[Structure]:
        """
        Apply strain strategy to the input structure.

        Args:
            structure: Base structure.
            strain_type: Type of deformation.
            magnitude: Max strain amplitude (e.g. 0.05 for 5%).
            n_points: Number of structures to generate (interpolating from -mag to +mag or 0 to mag).

        Returns:
            List of strained Structures.
        """
        atoms = structure.to_ase()
        results = []

        # Generate range of strains
        # If n_points is 1, use +magnitude.
        # If n_points > 1, linspace from -magnitude to +magnitude (excluding 0 if possible to avoid dupes)
        strains: np.ndarray | List[float]
        if n_points == 1:
            strains = [magnitude]
        else:
            strains = np.linspace(-magnitude, magnitude, n_points)
            # Remove near-zero if we want to avoid reproducing the original exactly?
            # It's fine to include it as a reference, but active learning prefers diversity.

        if strain_type == "rattle":
            # Rattle is different: it applies random noise to positions, not cell.
            builder = StructureBuilder()
            for _ in range(n_points):
                # Magnitude interpreted as stdev for rattle
                s = builder.apply_rattle(structure, stdev=magnitude)
                results.append(s)
            return results

        original_cell = atoms.get_cell() # type: ignore[no-untyped-call]

        for eps in strains:
            if abs(eps) < 1e-6:
                continue # Skip effectively zero strain

            new_atoms = atoms.copy() # type: ignore[no-untyped-call]
            cell = original_cell.copy()
            deformation = np.eye(3)

            if strain_type == "uniaxial":
                # Strain along x, y, or z. Let's do all 3 or random?
                # For simplicity, let's do X-axis or iterate.
                # To be robust, let's return variations for x, y, z if n_points is small.
                # Here we just do X-axis stretch for simplicity of this function,
                # or uniform expansion if "volumetric".
                deformation[0, 0] += eps

            elif strain_type == "volumetric":
                deformation[0, 0] += eps
                deformation[1, 1] += eps
                deformation[2, 2] += eps

            elif strain_type == "shear":
                # Apply shear (e.g. xy)
                # deformation = [[1, eps, 0], [0, 1, 0], [0, 0, 1]]
                deformation[0, 1] += eps
                # To preserve volume to 1st order, some shears are volume conserving.
                # Simple shear adds strain component.

            # Apply deformation: new_cell = cell @ deformation
            # Note: ASE set_cell with scale_atoms=True handles this but usually takes new cell parms.
            # We construct the new cell matrix.

            # cell is typically row vectors in ASE (3x3).
            # new_cell = cell * deformation ?
            # Let's verify: Vector v_new = v_old @ deformation.
            new_cell = np.dot(cell, deformation)

            new_atoms.set_cell(new_cell, scale_atoms=True) # type: ignore[no-untyped-call]
            results.append(Structure.from_ase(new_atoms))

        return results
