import logging
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class ElasticCalc:
    """
    Calculates Elastic Constants (Cij) and checks Born stability.
    """

    def __init__(self, strain_magnitude: float = 0.01) -> None:
        self.strain_magnitude = strain_magnitude

    def calculate(
        self, structure: Structure, calculator: Calculator, workdir: Path
    ) -> tuple[bool, float | None, float | None]:
        """
        Compute elastic constants and check stability.

        Args:
            structure: Equilibrium structure.
            calculator: ASE calculator.
            workdir: Directory for calculations.

        Returns:
            (is_stable, bulk_modulus, shear_modulus)
        """
        workdir.mkdir(parents=True, exist_ok=True)
        atoms_ref = structure.to_ase()

        C = np.zeros((6, 6))

        for j in range(6):
            # Apply +delta and -delta
            strains = [self.strain_magnitude, -self.strain_magnitude]
            stresses = []

            for delta in strains:
                atoms = atoms_ref.copy() # type: ignore[no-untyped-call]
                atoms.calc = calculator

                # Create strain tensor
                strain_voigt = np.zeros(6)
                strain_voigt[j] = delta

                e1, e2, e3, e4, e5, e6 = strain_voigt
                epsilon = np.array([
                    [e1, e6/2, e5/2],
                    [e6/2, e2, e4/2],
                    [e5/2, e4/2, e3]
                ])

                # Deform cell
                deformation = np.eye(3) + epsilon
                atoms.set_cell(atoms.cell @ deformation, scale_atoms=True) # type: ignore[no-untyped-call]

                # Calc stress
                try:
                    s = atoms.get_stress(voigt=True) # type: ignore[no-untyped-call]
                    stresses.append(s)
                except Exception:
                    logger.exception(f"Elastic calc failed for strain {j} delta {delta}")
                    return False, None, None

            # Centered difference
            diff = (stresses[0] - stresses[1]) / (2 * self.strain_magnitude)
            C[:, j] = diff

        # Check stability: C must be positive definite
        is_stable: bool
        try:
            eigvals = np.linalg.eigvalsh(C)
            # Ensure native bool type
            is_stable = bool(np.all(eigvals > 0))
        except Exception:
            is_stable = False

        # Calculate Bulk Modulus (Voigt average)
        B_V = ((C[0,0]+C[1,1]+C[2,2]) + 2*(C[0,1]+C[1,2]+C[0,2])) / 9.0

        # Shear Modulus (Voigt average)
        G_V = ((C[0,0]+C[1,1]+C[2,2]) - (C[0,1]+C[1,2]+C[0,2]) + 3*(C[3,3]+C[4,4]+C[5,5])) / 15.0

        # Reuss average (requires compliance S = inv(C))
        B_GPa: float | None
        G_GPa: float | None

        try:
            S = np.linalg.inv(C)
            B_R = 1.0 / ((S[0,0]+S[1,1]+S[2,2]) + 2*(S[0,1]+S[1,2]+S[0,2]))
            G_R = 15.0 / (4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[1,2]+S[0,2]) + 3*(S[3,3]+S[4,4]+S[5,5]))

            # Hill average
            B = (B_V + B_R) / 2.0
            G = (G_V + G_R) / 2.0

            # Convert to GPa (stress in ASE is eV/A^3 = 160.2 GPa)
            B_GPa = float(B * 160.21766208)
            G_GPa = float(G * 160.21766208)

        except Exception as e:
            logger.warning(f"Failed to invert C matrix: {e}")
            # Fallback to Voigt
            B_GPa = float(B_V * 160.21766208)
            G_GPa = float(G_V * 160.21766208)

        return is_stable, B_GPa, G_GPa
