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

        # We need equilibrium stress to be close to zero, but we take finite difference.
        # C_ij = d(sigma_i)/d(epsilon_j)

        # Voigt notation mapping for strains:
        # 0: xx, 1: yy, 2: zz, 3: yz, 4: xz, 5: xy
        # Strain matrix construction from vector e (6,)
        # e = [e1, e2, e3, e4, e5, e6]
        # epsilon = [[e1, e6/2, e5/2], [e6/2, e2, e4/2], [e5/2, e4/2, e3]]

        C = np.zeros((6, 6))

        for j in range(6):
            # Apply +delta and -delta
            strains = [self.strain_magnitude, -self.strain_magnitude]
            stresses = []

            for delta in strains:
                atoms = atoms_ref.copy()
                atoms.calc = calculator

                # Create strain tensor
                strain_voigt = np.zeros(6)
                strain_voigt[j] = delta

                # Convert to matrix
                # Note: Engineering strain vs Tensor strain.
                # ASE usually works with deformation gradient F.
                # F = I + epsilon
                # But for finite strain, we use: cell_new = cell_old @ (I + epsilon).
                # epsilon is the strain tensor.
                # Voigt vector: e1=exx, e2=eyy, e3=ezz, e4=2eyz, e5=2exz, e6=2exy
                # So e_tensor components:
                e1, e2, e3, e4, e5, e6 = strain_voigt
                epsilon = np.array([
                    [e1, e6/2, e5/2],
                    [e6/2, e2, e4/2],
                    [e5/2, e4/2, e3]
                ])

                # Deform cell
                # new_cell = old_cell @ (I + epsilon) ? No.
                # new_cell = (I + epsilon) @ old_cell if column vectors.
                # ASE uses row vectors for cell. So new_cell = old_cell @ (I + epsilon).T
                # Actually, correct is: new_cell = old_cell @ (I + epsilon)
                deformation = np.eye(3) + epsilon
                atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)

                # Calc stress
                # stress usually returned as [sxx, syy, szz, syz, sxz, sxy] (Voigt)
                try:
                    s = atoms.get_stress(voigt=True)
                    stresses.append(s)
                except Exception as e:
                    logger.error(f"Elastic calc failed for strain {j} delta {delta}: {e}")
                    return False, None, None

            # Centered difference
            # d(sigma)/d(epsilon) ~ (sigma(+) - sigma(-)) / (2*delta)
            # Note: Stresses are [s1, s2, s3, s4, s5, s6]
            diff = (stresses[0] - stresses[1]) / (2 * self.strain_magnitude)
            C[:, j] = diff

        # Check stability: C must be positive definite
        try:
            eigvals = np.linalg.eigvalsh(C)
            is_stable = np.all(eigvals > 0)
        except Exception:
            is_stable = False

        # Calculate Bulk Modulus (Voigt average)
        # B_V = ( (C11+C22+C33) + 2(C12+C23+C13) ) / 9
        B_V = ((C[0,0]+C[1,1]+C[2,2]) + 2*(C[0,1]+C[1,2]+C[0,2])) / 9.0

        # Shear Modulus (Voigt average)
        # G_V = ( (C11+C22+C33) - (C12+C23+C13) + 3(C44+C55+C66) ) / 15
        G_V = ((C[0,0]+C[1,1]+C[2,2]) - (C[0,1]+C[1,2]+C[0,2]) + 3*(C[3,3]+C[4,4]+C[5,5])) / 15.0

        # Reuss average (requires compliance S = inv(C))
        try:
            S = np.linalg.inv(C)
            # B_R = 1 / [ (S11+S22+S33) + 2(S12+S23+S13) ]
            B_R = 1.0 / ((S[0,0]+S[1,1]+S[2,2]) + 2*(S[0,1]+S[1,2]+S[0,2]))

            # G_R = 15 / [ 4(S11+S22+S33) - 4(S12+S23+S13) + 3(S44+S55+S66) ]
            G_R = 15.0 / (4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[1,2]+S[0,2]) + 3*(S[3,3]+S[4,4]+S[5,5]))

            # Hill average
            B = (B_V + B_R) / 2.0
            G = (G_V + G_R) / 2.0

            # Convert to GPa (stress in ASE is eV/A^3 = 160.2 GPa)
            B_GPa = B * 160.21766208
            G_GPa = G * 160.21766208

            return is_stable, B_GPa, G_GPa

        except Exception as e:
            logger.warning(f"Failed to invert C matrix: {e}")
            # Fallback to Voigt
            B_GPa = B_V * 160.21766208
            G_GPa = G_V * 160.21766208
            return is_stable, B_GPa, G_GPa
