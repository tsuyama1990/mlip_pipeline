import numpy as np
from ase import Atoms
from ase.units import GPa

from mlip_autopipec.config.schemas.validation import ElasticConfig
from mlip_autopipec.data_models.validation import ValidationMetric, ValidationResult


class ElasticityValidator:
    """
    Validates elastic stability by calculating the stiffness matrix C_ij.
    """

    def __init__(self, config: ElasticConfig) -> None:
        self.config = config

    def validate(self, atoms: Atoms) -> ValidationResult:
        """
        Calculate elastic tensor and check stability conditions.

        Args:
            atoms: ASE Atoms object with calculator.

        Returns:
            ValidationResult with C_ij and stability check.
        """
        if atoms.calc is None:
            return ValidationResult(
                module="elastic", passed=False, error="Atoms object has no calculator attached."
            )

        try:
            # 1. Calculate Stiffness Matrix (6x6)
            # Unit: GPa (typical output)
            C_matrix = self._calculate_stiffness_matrix(atoms)

            # 2. Check Stability
            # A crystal is mechanically stable if the stiffness matrix is positive definite.
            # This is a necessary and sufficient condition for any crystal system.
            eigenvalues = np.linalg.eigvalsh(C_matrix)
            min_eig = float(np.min(eigenvalues))

            # Allow small numerical noise around zero? No, strictly positive for stability.
            # But usually we define a threshold.
            passed = min_eig > 0.0

            metrics = [
                ValidationMetric(
                    name="min_eigenvalue",
                    value=min_eig,
                    unit="GPa",
                    passed=passed
                ),
                ValidationMetric(
                    name="C_matrix_flattened",
                    value=C_matrix.flatten().tolist(), # Store as list
                    unit="GPa",
                    passed=True,
                    details={"shape": [6, 6]}
                )
            ]

            return ValidationResult(
                module="elastic",
                passed=passed,
                metrics=metrics
            )

        except Exception as e:
            return ValidationResult(
                module="elastic",
                passed=False,
                error=f"Elasticity calculation failed: {e!s}"
            )

    def _calculate_stiffness_matrix(self, atoms: Atoms) -> np.ndarray:
        """
        Calculate 6x6 stiffness matrix using finite differences of stress vs strain.
        """
        C = np.zeros((6, 6))

        # Standard procedure:
        # For each strain component i in 0..5 (xx, yy, zz, yz, xz, xy):
        #   Apply +/- delta strain
        #   Measure stress sigma
        #   C_ji = d_sigma_j / d_epsilon_i

        delta = self.config.max_distortion

        for i in range(6):
            stresses = []
            for d in [-delta, delta]:
                temp_atoms = atoms.copy()
                temp_atoms.calc = atoms.calc

                # Create strain matrix
                strain = np.zeros(6)
                strain[i] = d

                # Apply strain
                strain_tensor = np.zeros((3, 3))
                strain_tensor[0, 0] = strain[0]
                strain_tensor[1, 1] = strain[1]
                strain_tensor[2, 2] = strain[2]

                # Voigt: 3=yz, 4=xz, 5=xy
                strain_tensor[1, 2] = strain[3] / 2.0
                strain_tensor[2, 1] = strain[3] / 2.0
                strain_tensor[0, 2] = strain[4] / 2.0
                strain_tensor[2, 0] = strain[4] / 2.0
                strain_tensor[0, 1] = strain[5] / 2.0
                strain_tensor[1, 0] = strain[5] / 2.0

                # New cell = (I + strain) * old_cell
                deformation = np.eye(3) + strain_tensor
                new_cell = np.dot(atoms.cell, deformation)

                temp_atoms.set_cell(new_cell, scale_atoms=True)

                # Get stress
                stress = temp_atoms.get_stress()
                # ASE stress is usually in eV/A^3. Convert to GPa.
                stress_gpa = stress / GPa
                stresses.append(stress_gpa)

            # Finite difference
            # C_ji = (sigma_j(+) - sigma_j(-)) / (2 * delta)
            ds = (stresses[1] - stresses[0]) / (2 * delta)
            C[:, i] = ds

        return C
