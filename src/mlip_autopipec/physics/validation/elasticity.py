from pathlib import Path
import numpy as np
from ase.build import bulk

from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.runner import BaseValidator
from mlip_autopipec.physics.validation.utils import get_calculator


class ElasticityValidator(BaseValidator):
    def validate(self, potential_path: Path) -> ValidationResult:
        element = self.potential_config.elements[0]
        try:
            atoms = bulk(element) # type: ignore[no-untyped-call]
        except Exception:
            atoms = bulk(element, 'fcc', a=4.0) # type: ignore[no-untyped-call]

        calc = get_calculator(potential_path, self.potential_config)

        C = self._calculate_cij(atoms, calc)

        # Check stability for cubic (simplified check, assuming cubic structure)
        c11 = C[0,0]
        c12 = C[0,1]
        c44 = C[3,3]

        passed = True

        checks = [
            ("C11 - C12 > 0", c11 - c12 > 0),
            ("C11 + 2C12 > 0", c11 + 2*c12 > 0),
            ("C44 > 0", c44 > 0)
        ]

        failed_checks = [name for name, res in checks if not res]
        if failed_checks:
            passed = False

        metrics = [
            ValidationMetric(name="C11", value=float(c11), passed=True),
            ValidationMetric(name="C12", value=float(c12), passed=True),
            ValidationMetric(name="C44", value=float(c44), passed=True),
            ValidationMetric(name="Stability", value=1.0 if passed else 0.0, passed=passed)
        ]

        return ValidationResult(
            potential_id=potential_path.stem,
            metrics=metrics,
            plots={},
            overall_status="PASS" if passed else "FAIL"
        )

    def _calculate_cij(self, atoms, calc) -> np.ndarray:
        delta = self.config.strain_magnitude

        # Voigt notation 6 components
        C = np.zeros((6,6))

        def get_stress(strain_voigt):
             strain_tensor = np.zeros((3,3))
             strain_tensor[0,0] = strain_voigt[0]
             strain_tensor[1,1] = strain_voigt[1]
             strain_tensor[2,2] = strain_voigt[2]

             # Off-diagonal: Voigt definition 2*epsilon_ij = gamma_ij
             strain_tensor[1,2] = strain_tensor[2,1] = strain_voigt[3] / 2.0
             strain_tensor[0,2] = strain_tensor[2,0] = strain_voigt[4] / 2.0
             strain_tensor[0,1] = strain_tensor[1,0] = strain_voigt[5] / 2.0

             deformation = np.eye(3) + strain_tensor

             at = atoms.copy() # type: ignore[no-untyped-call]
             # ASE cell is row-vectors. cell_new = cell_old @ deformation_T?
             # If r' = F r.
             # h = [a, b, c]^T (column vectors in standard physics).
             # ASE h_ase = h^T.
             # h_new = F h.
             # h_ase_new = h_new^T = (F h)^T = h^T F^T.
             # So cell_new = cell @ F^T.
             # If deformation IS F.
             # My deformation matrix above is F.
             at.set_cell(at.cell @ deformation.T, scale_atoms=True)

             at.calc = calc

             # get_stress returns [sxx, syy, szz, syz, sxz, sxy] (Voigt order)
             return at.get_stress(voigt=True)

        # Reference (not strictly needed if using central difference centered at 0)

        for i in range(6):
            strain = np.zeros(6)
            strain[i] = delta
            stress_plus = get_stress(strain)

            strain[i] = -delta
            stress_minus = get_stress(strain)

            # C_ij = d(sigma_i)/d(epsilon_j)
            # We varied epsilon_i (loop variable), so we get column i of C
            C[:, i] = (stress_plus - stress_minus) / (2 * delta)

        # Convert to GPa
        C_GPa = C * 160.21766208
        return C_GPa
