from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from .utils import get_lammps_calculator


class ElasticityValidator:
    def __init__(self, config: ValidationConfig, potential_path: Path):
        self.config = config
        self.potential_path = potential_path

    def validate(self, structure: Structure) -> ValidationResult:
        try:
            C = self._calculate_stiffness(structure)

            # Check cubic stability
            # This logic assumes cubic crystal structure
            # TODO: Generalize for other crystal systems
            c11 = C[0, 0]
            c12 = C[0, 1]
            c44 = C[3, 3]

            # Born criteria for cubic
            cond1 = (c11 - c12) > 0
            cond2 = (c11 + 2 * c12) > 0
            cond3 = c44 > 0

            passed = cond1 and cond2 and cond3

            # Helper to create metric safely
            def mk_metric(name, val, pass_cond):
                return ValidationMetric(name=name, value=float(val), passed=pass_cond)

            metrics = [
                mk_metric("C11", c11, True),
                mk_metric("C12", c12, True),
                mk_metric("C44", c44, c44 > 0),
                ValidationMetric(
                    name="Born Stability",
                    value=1.0 if passed else 0.0,
                    passed=bool(passed),
                ),
            ]

            return ValidationResult(
                potential_id=str(self.potential_path),
                metrics=metrics,
                plots={},
                overall_status="PASS" if passed else "FAIL",
            )

        except Exception as e:
            # We must construct a ValidationResult even on error
            # To pass schema, we need dummy values or handle it properly
            # We can put error in value as float (NaN?) or just fail
            return ValidationResult(
                potential_id=str(self.potential_path),
                metrics=[
                    ValidationMetric(
                        name="Error", value=0.0, message=str(e), passed=False
                    )
                ],
                plots={},
                overall_status="FAIL",
            )

    def _calculate_stiffness(self, structure: Structure) -> np.ndarray:
        atoms = structure.to_ase()
        elements = sorted(list(set(atoms.get_chemical_symbols())))
        calc = get_lammps_calculator(self.potential_path, elements)
        atoms.calc = calc

        delta = 0.01  # 1% strain

        # 1. Strain xx
        atoms_xx = atoms.copy()  # type: ignore[no-untyped-call]
        cell = atoms.get_cell()
        # Strain tensor
        eps = np.eye(3)
        eps[0, 0] += delta
        atoms_xx.set_cell(cell @ eps, scale_atoms=True)
        stress_xx = atoms_xx.get_stress()  # [xx, yy, zz, yz, xz, xy]

        c11 = stress_xx[0] / delta
        c12 = stress_xx[1] / delta

        # 2. Strain xy (Shear)
        atoms_xy = atoms.copy()  # type: ignore[no-untyped-call]
        eps = np.eye(3)
        # Symmetrized shear strain
        eps[0, 1] += delta / 2
        eps[1, 0] += delta / 2
        atoms_xy.set_cell(cell @ eps, scale_atoms=True)
        stress_xy = atoms_xy.get_stress()

        c44 = stress_xy[5] / delta

        # Construct matrix (partial)
        # Convert to GPa (1 eV/A^3 = 160.2 GPa)
        conv = 160.21766208
        C = np.zeros((6, 6))
        C[0, 0] = c11 * conv
        C[0, 1] = c12 * conv
        C[3, 3] = c44 * conv

        return C
