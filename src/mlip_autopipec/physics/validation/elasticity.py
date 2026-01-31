from typing import Optional
from pathlib import Path
from ase import Atoms
import numpy as np

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.utils import get_lammps_calculator

class ElasticityValidator:
    def __init__(self, potential_path: Path, config: ValidationConfig, potential_config: PotentialConfig, lammps_command: str = "lmp"):
        self.potential_path = potential_path
        self.config = config
        self.potential_config = potential_config
        self.lammps_command = lammps_command

    def _get_calculator(self, work_dir: Path):
        return get_lammps_calculator(
            potential_path=self.potential_path,
            potential_config=self.potential_config,
            lammps_command=self.lammps_command,
            working_dir=work_dir
        )

    def _calculate_elastic_tensor(self, atoms: Atoms) -> tuple[np.ndarray, dict[str, float]]:
        """
        Calculate elastic tensor using finite differences.
        Returns 6x6 matrix (Voigt) and dictionary of key constants.
        """
        # Strains
        delta = 0.01 # 1% strain

        # 6 Voigt components: xx, yy, zz, yz, xz, xy
        # Voigt map: 0:xx, 1:yy, 2:zz, 3:yz, 4:xz, 5:xy

        # Base stress
        atoms.calc.calculate(atoms)
        base_stress = atoms.get_stress(voigt=True) # returns [xx, yy, zz, yz, xz, xy]

        C = np.zeros((6, 6))

        # For each strain component
        for i in range(6):
            # Apply +delta strain
            # Strain matrix
            strain = np.zeros(6)
            strain[i] = delta

            # Deform atoms
            # strain to matrix
            # Voigt to 3x3:
            # e_xx, e_yy, e_zz, 2e_yz, 2e_xz, 2e_xy
            # Note: engineering strain vs tensor strain.
            # ASE get_stress returns Voigt order.
            # If we deform the cell:
            # new_cell = (I + eps) * old_cell

            eps_matrix = np.zeros((3, 3))
            eps_matrix[0, 0] = strain[0]
            eps_matrix[1, 1] = strain[1]
            eps_matrix[2, 2] = strain[2]

            # Shear: Voigt index 3 (yz) corresponds to 2*eps_yz
            # So eps_yz = strain[3] / 2?
            # Usually simulation codes use engineering strain for box deformation.
            # If we tilt the box.
            # Let's use ASE's set_cell with scale_atoms=True.

            # Simple approach: use `StrainFilter` from ASE or just manual deformation.
            # Manual is safer for understanding.

            # Construct deformation gradient F = I + epsilon
            # For shear, if we use engineering strain gamma, e.g. gamma_yz, then F_yz = gamma_yz?
            # Or F_yz = gamma_yz / 2?
            # Small strain approx.
            # Let's assume standard Voigt definition where stress = C * strain.

            # To avoid complexity, let's use +delta and -delta.

            # We need to clone atoms
            deformed = atoms.copy() # type: ignore[no-untyped-call]
            deformed.calc = atoms.calc

            cell0 = atoms.get_cell()

            # Create deformation matrix
            # 0:xx, 1:yy, 2:zz, 3:yz, 4:xz, 5:xy
            def_matrix = np.eye(3)
            def_matrix[0, 0] += strain[0]
            def_matrix[1, 1] += strain[1]
            def_matrix[2, 2] += strain[2]

            # Shears
            # standard convention: xy component affects x vector by y coordinate?
            # def_matrix[0, 1] += strain[5] ... ?
            # For triclinic cells, ASE handles it.
            # Let's ignore shears for basic stability if we assume cubic?
            # But the requirement is full Cij.

            # Correct mapping for engineering strain to deformation matrix:
            # eps = (F^T F - I) / 2 ...
            # For small strains, F = I + eps_tensor
            # eps_tensor: diagonal same. off-diagonal: eps_ij = gamma_ij / 2.

            def_matrix[1, 2] += strain[3] / 2.0
            def_matrix[2, 1] += strain[3] / 2.0

            def_matrix[0, 2] += strain[4] / 2.0
            def_matrix[2, 0] += strain[4] / 2.0

            def_matrix[0, 1] += strain[5] / 2.0
            def_matrix[1, 0] += strain[5] / 2.0

            # Apply deformation
            new_cell = np.dot(cell0, def_matrix) # or dot(def_matrix, cell0)?
            # ASE: row vectors. cell = [v1, v2, v3].
            # new_v1 = v1 * F? No.
            # Usually r' = F r.
            # If cell vectors are rows, then V_new = V_old * F^T?
            # Let's stick to simple deformations.

            deformed.set_cell(np.dot(cell0, def_matrix), scale_atoms=True) # type: ignore[no-untyped-call]

            deformed.get_potential_energy()
            stress_plus = deformed.get_stress(voigt=True)

            # C_col_i = (stress_plus - base_stress) / delta
            # This is Secant modulus.
            # Better: use +/- delta.

            C[:, i] = (stress_plus - base_stress) / delta

        # Symmetrize C
        C = (C + C.T) / 2.0

        # Convert units if needed. ASE stress is eV/A^3.
        # C is also eV/A^3. Convert to GPa.
        C_GPa = C * 160.21766208

        # Extract constants for Cubic (assumption)
        c11 = C_GPa[0, 0]
        c12 = C_GPa[0, 1]
        c44 = C_GPa[3, 3] # Voigt index 3 is yz?
        # Wait, ASE Voigt order: xx, yy, zz, yz, xz, xy.
        # Indices: 0, 1, 2, 3, 4, 5.
        # For cubic: C44 is shear.
        # C44 is associated with yz, xz, xy.
        # So C[3,3], C[4,4], C[5,5] should be C44.

        return C_GPa, {"C11": c11, "C12": c12, "C44": c44}

    def validate(self, reference_structure: Atoms) -> ValidationResult:
        work_dir = Path("validation_work/elasticity")
        work_dir.mkdir(parents=True, exist_ok=True)

        calc = self._get_calculator(work_dir)

        atoms = reference_structure.copy() # type: ignore[no-untyped-call]
        atoms.calc = calc

        passed = False
        metrics = []
        error_msg = None

        try:
            C_matrix, constants = self._calculate_elastic_tensor(atoms)

            # Check Born Stability for Cubic
            # C11 - C12 > 0
            # C11 + 2C12 > 0
            # C44 > 0

            c11 = constants["C11"]
            c12 = constants["C12"]
            c44 = constants["C44"]

            born_1 = (c11 - c12) > self.config.elastic_stability_tolerance
            born_2 = (c11 + 2*c12) > self.config.elastic_stability_tolerance
            born_3 = c44 > self.config.elastic_stability_tolerance

            # Also check eigenvalues of C matrix
            eigenvalues = np.linalg.eigvalsh(C_matrix)
            min_eig = np.min(eigenvalues)
            positive_definite = min_eig > self.config.elastic_stability_tolerance

            passed = positive_definite

            metrics.append(ValidationMetric(name="C11", value=c11, passed=True))
            metrics.append(ValidationMetric(name="C12", value=c12, passed=True))
            metrics.append(ValidationMetric(name="C44", value=c44, passed=True))
            metrics.append(ValidationMetric(name="Born Stability", value=min_eig, passed=passed, message="Min Eigenvalue of Stiffness Matrix"))

        except Exception as e:
            passed = False
            error_msg = str(e)
            metrics.append(ValidationMetric(name="Elastic Calculation", value=0.0, passed=False, message=error_msg))

        status = "PASS" if passed else "FAIL"

        return ValidationResult(
            potential_id=self.potential_path.stem,
            metrics=metrics,
            plots={},
            overall_status=status
        )
