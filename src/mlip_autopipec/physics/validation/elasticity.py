from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.physics.validation.base import BaseValidator


class ElasticityValidator(BaseValidator):
    def __init__(
        self,
        structure: Atoms,
        calculator: Calculator,
        config: ValidationConfig,
        work_dir: Path,
        potential_id: str,
    ):
        self.structure = structure.copy()  # type: ignore[no-untyped-call]
        self.calculator = calculator
        self.config = config
        self.work_dir = work_dir
        self.potential_id = potential_id
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> ValidationResult:
        delta = self.config.elastic_strain
        C = np.zeros((6, 6))

        # Voigt mapping for ASE stress order: xx, yy, zz, yz, xz, xy
        # This matches standard Voigt index 0..5

        original_cell = self.structure.get_cell()

        try:
            for j in range(6):
                # Calculate derivative d(sigma)/d(strain_j) using central difference

                # Positive strain
                strain_plus = np.zeros(6)
                strain_plus[j] = delta
                atoms_plus = self._apply_strain(
                    self.structure, strain_plus, original_cell
                )
                atoms_plus.calc = self.calculator
                stress_plus = atoms_plus.get_stress()  # Voigt form

                # Negative strain
                strain_minus = np.zeros(6)
                strain_minus[j] = -delta
                atoms_minus = self._apply_strain(
                    self.structure, strain_minus, original_cell
                )
                atoms_minus.calc = self.calculator
                stress_minus = atoms_minus.get_stress()

                # C_ij = d(sigma_i) / d(strain_j)
                # Note: ASE stress is in eV/A^3. We want GPa.
                # 1 eV/A^3 = 160.21766208 GPa
                factor = 160.21766208

                diff_stress = (stress_plus - stress_minus) * factor
                C[:, j] = diff_stress / (2 * delta)

            # Symmetrize
            C = (C + C.T) / 2.0

            # Check Born stability (Positive Definiteness)
            eigvals = np.linalg.eigvalsh(C)
            min_eig = np.min(eigvals)
            passed = min_eig > 0

            # Generate Heatmap
            plot_path = self.work_dir / "elasticity_matrix.png"
            self._plot_matrix(C, plot_path)

            metric = ValidationMetric(
                name="Elastic Stability (Min Eigenvalue)",
                value=float(min_eig),
                passed=passed,
            )

            # Also add Bulk Modulus from C?
            # For cubic: B = (C11 + 2C12)/3. But general case?
            # Voigt average B_v = ((C11+C22+C33) + 2(C12+C23+C13))/9
            # Let's verify Voigt average formula.
            # B_v = 1/9 * sum(C_ij) for i,j in {0,1,2}? No.
            # It's (Tr(C_upper_left) + 2 * sum(off_diagonals_upper_left))/9

            c11, c22, c33 = C[0, 0], C[1, 1], C[2, 2]
            c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]
            b_modulus = (c11 + c22 + c33 + 2 * (c12 + c13 + c23)) / 9.0

            metric_b = ValidationMetric(
                name="Bulk Modulus (Elastic)",
                value=float(b_modulus),
                passed=passed,  # If stable, B should be positive too
            )

            return ValidationResult(
                potential_id=self.potential_id,
                metrics=[metric, metric_b],
                plots={"Elasticity Matrix": plot_path},
                overall_status="PASS" if passed else "FAIL",
            )

        except Exception as e:
            return ValidationResult(
                potential_id=self.potential_id,
                metrics=[
                    ValidationMetric(
                        name="Elasticity", value=0.0, passed=False, error_message=str(e)
                    )
                ],
                overall_status="FAIL",
            )

    def _apply_strain(
        self, structure: Atoms, voigt_strain: np.ndarray, original_cell: np.ndarray
    ) -> Atoms:
        """
        Apply Voigt strain to structure.
        voigt_strain: [e_xx, e_yy, e_zz, g_yz, g_xz, g_xy]
        Note: g are engineering shear strains (= 2 * epsilon).
        """
        e = voigt_strain
        # Strain tensor epsilon
        # e[3] = 2 * eps_yz => eps_yz = e[3]/2

        eps = np.array(
            [
                [e[0], e[5] / 2, e[4] / 2],
                [e[5] / 2, e[1], e[3] / 2],
                [e[4] / 2, e[3] / 2, e[2]],
            ]
        )

        # Deformation matrix F = I + eps (small strain approx)
        # Or better: h_new = h_old @ (I + eps)
        # Check multiplication order.
        # If cell vectors are rows: v' = v (I + eps)

        deformation = np.eye(3) + eps
        new_cell = original_cell @ deformation

        atoms = structure.copy()  # type: ignore[no-untyped-call]
        atoms.set_cell(new_cell, scale_atoms=True)
        return atoms

    def _plot_matrix(self, C: np.ndarray, path: Path):
        try:
            fig, ax = plt.subplots()
            cax = ax.matshow(C, cmap="viridis")
            fig.colorbar(cax)
            ax.set_title("Elastic Stiffness Matrix C (GPa)")
            # Add text
            for i in range(6):
                for j in range(6):
                    ax.text(
                        j,
                        i,
                        f"{C[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=8,
                    )
            fig.savefig(path)
            plt.close(fig)
        except Exception:
            pass
