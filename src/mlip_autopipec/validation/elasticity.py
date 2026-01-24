import logging
from typing import Any

import numpy as np
from ase import Atoms
from ase.units import GPa

from mlip_autopipec.config.schemas.validation import ElasticityConfig

logger = logging.getLogger(__name__)


class ElasticityValidator:
    """
    Validates the elastic stability (Born criteria) of a structure.
    Calculates the 6x6 elastic tensor Cij using stress-strain method.
    """

    def __init__(self, config: ElasticityConfig) -> None:
        self.config = config

    def validate(self, atoms: Atoms, calculator: Any) -> bool:
        """
        Runs elasticity calculation and checks Born stability.

        Args:
            atoms: The structure.
            calculator: ASE calculator.

        Returns:
            bool: True if stable.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected ase.Atoms object, got {type(atoms)}")

        try:
            C = self.calculate_elastic_tensor(atoms, calculator)
        except Exception as e:
            logger.error(f"Elasticity calculation failed: {e}")
            return False

        # Log Cij
        logger.info(f"Elastic Tensor Cij (GPa):\n{C}")

        # Detect system type. For this implementation, we default to cubic checks
        # or positive definiteness which is a general condition for stability.
        # General stability: C must be positive definite.

        # We'll use positive definite check as the most general "stable" criteria
        # alongside specific cubic checks if possible.
        is_pd = self.check_positive_definite(C)
        if not is_pd:
            logger.warning("Elastic tensor is not positive definite.")

        # Also check cubic criteria explicitly as they are common and requested
        is_cubic_stable = self.check_born_stability(C, "cubic")

        return is_pd and is_cubic_stable

    def calculate_elastic_tensor(self, atoms: Atoms, calculator: Any) -> np.ndarray:
        """
        Calculates 6x6 elastic tensor Cij (in GPa).
        """
        n_strains = 6
        strain_amount = self.config.strain_max  # e.g. 0.01

        # Use +strain and -strain for central difference?
        # Or just +strain. Central difference is better.

        C = np.zeros((6, 6))
        volume = atoms.get_volume()

        base_cell = atoms.get_cell()

        # Voigt notation indices: 0:xx, 1:yy, 2:zz, 3:yz, 4:xz, 5:xy

        for i in range(6):
            # Apply +delta and -delta
            stresses = []
            for sign in [1.0, -1.0]:
                delta = sign * strain_amount
                strain_voigt = np.zeros(6)
                strain_voigt[i] = delta

                # Construct strain matrix
                # e_matrix = [[e0, e5/2, e4/2], [e5/2, e1, e3/2], [e4/2, e3/2, e2]]
                e_matrix = np.array(
                    [
                        [strain_voigt[0], strain_voigt[5] / 2.0, strain_voigt[4] / 2.0],
                        [strain_voigt[5] / 2.0, strain_voigt[1], strain_voigt[3] / 2.0],
                        [strain_voigt[4] / 2.0, strain_voigt[3] / 2.0, strain_voigt[2]],
                    ]
                )

                deformation = np.eye(3) + e_matrix
                new_cell = np.dot(base_cell, deformation)

                # Create deformed atoms
                deformed = atoms.copy()  # type: ignore[no-untyped-call]
                deformed.set_cell(new_cell, scale_atoms=True)
                deformed.calc = calculator

                # Get stress (Voigt form: xx, yy, zz, yz, xz, xy)
                # ASE get_stress returns [sxx, syy, szz, syz, sxz, sxy]
                # Default is eV/A^3.
                stress = deformed.get_stress(voigt=True)
                stresses.append(stress)

            # Central difference: dSigma/dEpsilon ~= (Sigma(+) - Sigma(-)) / (2*delta)
            # This gives the i-th column of C
            slope = (stresses[0] - stresses[1]) / (2 * strain_amount)

            # C_ji = slope_j (Change in stress component j due to strain i)
            C[:, i] = slope

        # Convert to GPa
        C_GPa = C / GPa
        return C_GPa

    @staticmethod
    def check_positive_definite(C: np.ndarray) -> bool:
        try:
            # Check if all eigenvalues are positive
            eigvals = np.linalg.eigvalsh(C)
            min_eig = np.min(eigvals)
            logger.info(f"Elastic Tensor Min Eigenvalue: {min_eig}")
            return bool(min_eig > 0)
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def check_born_stability(C: np.ndarray, system_type: str = "cubic") -> bool:
        """
        Checks Born stability criteria.
        Currently supports cubic.
        """
        if system_type == "cubic":
            c11 = C[0, 0]
            c12 = C[0, 1]
            c44 = C[3, 3]

            cond1 = (c11 - c12) > 0
            cond2 = (c11 + 2 * c12) > 0
            cond3 = c44 > 0

            logger.info(
                f"Born Criteria (Cubic): C11-C12={c11 - c12:.2f}, C11+2C12={c11 + 2 * c12:.2f}, C44={c44:.2f}"
            )

            if not (cond1 and cond2 and cond3):
                logger.warning("Born stability criteria failed for cubic system.")
                return False
            return True

        # TODO: Implement other systems
        return True
