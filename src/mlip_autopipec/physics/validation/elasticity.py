from pathlib import Path

import numpy as np
from ase.units import GPa

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import (
    ValidationConfig,
    ValidationMetric,
)
from mlip_autopipec.physics.validation.utils import get_validation_calculator


class ElasticityValidator:
    def __init__(
        self,
        val_config: ValidationConfig,
        pot_config: PotentialConfig,
        work_dir: Path,
    ):
        self.val_config = val_config
        self.pot_config = pot_config
        self.work_dir = work_dir / "elasticity"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(
        self, structure: Structure, potential_path: Path
    ) -> tuple[list[ValidationMetric], dict[str, Path]]:
        atoms = structure.to_ase()
        calc = get_validation_calculator(potential_path, self.pot_config, self.work_dir)
        atoms.calc = calc

        # Calculate C matrix
        try:
            C = self.calculate_stiffness_matrix(atoms)

            # Check stability (Positive Definite)
            # Eigenvalues of C matrix must be positive
            eigvals = np.linalg.eigvalsh(C)
            min_eig = np.min(eigvals)
            passed = min_eig > 0

            metric = ValidationMetric(
                name="Elastic Stability",
                value=float(min_eig),
                passed=bool(passed),
                unit="GPa",
            )
            return [metric], {}
        except Exception:
            return [
                ValidationMetric(
                    name="Elastic Stability", value=0.0, passed=False, unit="GPa"
                )
            ], {}

    def calculate_stiffness_matrix(self, atoms) -> np.ndarray:  # type: ignore[no-untyped-def]
        # returns 6x6 matrix in GPa
        delta = self.val_config.elastic_strain_mag
        cell0 = atoms.get_cell()

        # Voigt indices: xx, yy, zz, yz, xz, xy
        C = np.zeros((6, 6))

        # Apply strains one by one
        # Use central difference: (Sigma(+) - Sigma(-)) / (2*delta)

        for i in range(6):
            # Construct strain matrix
            strain = np.zeros(6)

            # +delta
            strain[i] = delta
            atoms_plus = atoms.copy()  # type: ignore[no-untyped-call]
            atoms_plus.calc = atoms.calc
            cell_plus = self._deform_cell(cell0, strain)
            atoms_plus.set_cell(cell_plus, scale_atoms=True)  # type: ignore[no-untyped-call]
            # get_stress returns [xx, yy, zz, yz, xz, xy] in eV/A^3
            stress_plus = atoms_plus.get_stress(voigt=True)  # type: ignore[no-untyped-call]

            # -delta
            strain[i] = -delta
            atoms_minus = atoms.copy()  # type: ignore[no-untyped-call]
            atoms_minus.calc = atoms.calc
            cell_minus = self._deform_cell(cell0, strain)
            atoms_minus.set_cell(cell_minus, scale_atoms=True)  # type: ignore[no-untyped-call]
            stress_minus = atoms_minus.get_stress(voigt=True)  # type: ignore[no-untyped-call]

            # Derivative dSigma / dStrain
            deriv = (stress_plus - stress_minus) / (2 * delta)

            # Convert to GPa
            C[:, i] = deriv / GPa

        # Symmetrize C? Cij should be Cji.
        C = (C + C.T) / 2.0
        return C

    def _deform_cell(self, cell, voigt_strain):  # type: ignore[no-untyped-def]
        # Construct strain tensor from Voigt
        # e = [e1, e2, e3, e4, e5, e6] -> [xx, yy, zz, yz, xz, xy]
        # Engineering shear strain gamma = 2*epsilon
        e = voigt_strain
        eps = np.array(
            [
                [e[0], e[5] / 2, e[4] / 2],
                [e[5] / 2, e[1], e[3] / 2],
                [e[4] / 2, e[3] / 2, e[2]],
            ]
        )

        # New cell = (I + eps) * Old Cell
        # Cell vectors are rows in ASE
        deformation = np.eye(3) + eps
        return np.dot(cell, deformation.T)
