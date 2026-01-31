from pathlib import Path
import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric
from mlip_autopipec.physics.validation.utils import get_lammps_calculator

class ElasticityValidator:
    def __init__(self, val_config: ValidationConfig, pot_config: PotentialConfig, potential_path: Path, work_dir: Path = Path("_work_validation/elastic")):
        self.val_config = val_config
        self.pot_config = pot_config
        self.potential_path = potential_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, structure: Structure) -> ValidationMetric:
        # 1. Calculate Stiffness Matrix C_ij
        try:
            C_ij = self._calculate_stiffness(structure)
        except Exception as e:
            return ValidationMetric(
                name="Elastic Stability", value=0.0, passed=False, message=str(e)
            )

        # 2. Check Stability
        # General condition: positive definite C_ij
        try:
            eigvals = np.linalg.eigvalsh(C_ij)
            min_eig = np.min(eigvals)

            # Tolerance
            tol = self.val_config.elastic_stability_tolerance
            passed = min_eig > tol

            return ValidationMetric(
                name="Elastic Stability",
                value=min_eig,
                passed=bool(passed),
                message=f"Min Eigval: {min_eig:.4f} GPa. " + ("Stable" if passed else "Unstable")
            )
        except Exception as e:
            return ValidationMetric(
                name="Elastic Stability", value=0.0, passed=False, message=f"Check failed: {e}"
            )

    def _calculate_stiffness(self, structure: Structure) -> np.ndarray:
        atoms = structure.to_ase()

        calc = self._get_calculator()
        atoms.calc = calc

        delta = 0.01 # 1% strain
        C = np.zeros((6, 6))

        # Voigt notation: 0:xx, 1:yy, 2:zz, 3:yz, 4:xz, 5:xy

        for i in range(6):
            # Apply +delta strain
            atoms_plus = atoms.copy() # type: ignore[no-untyped-call]
            strain_plus = np.zeros(6)
            strain_plus[i] = delta
            self._apply_strain(atoms_plus, strain_plus)
            atoms_plus.calc = calc
            stress_plus = atoms_plus.get_stress()

            # Apply -delta strain
            atoms_minus = atoms.copy() # type: ignore[no-untyped-call]
            strain_minus = np.zeros(6)
            strain_minus[i] = -delta
            self._apply_strain(atoms_minus, strain_minus)
            atoms_minus.calc = calc
            stress_minus = atoms_minus.get_stress()

            # Central difference
            # Stress is returned as [xx, yy, zz, yz, xz, xy]
            diff = (stress_plus - stress_minus) / (2 * delta)
            C[:, i] = diff

        # Symmetrize C
        C = (C + C.T) / 2.0

        # Convert to GPa (1 eV/Ang^3 = 160.21766208 GPa)
        return C * 160.21766208

    def _apply_strain(self, atoms: Atoms, strain_voigt: np.ndarray) -> None:
        # Construct strain tensor
        e = strain_voigt
        epsilon = np.array([
            [e[0], 0.5*e[5], 0.5*e[4]],
            [0.5*e[5], e[1], 0.5*e[3]],
            [0.5*e[4], 0.5*e[3], e[2]]
        ])

        cell = atoms.get_cell()
        # cell is 3x3 array of row vectors
        # new_cell = cell * (I + epsilon)
        atoms.set_cell(np.dot(cell, np.eye(3) + epsilon), scale_atoms=True) # type: ignore[no-untyped-call]

    def _get_calculator(self):
        return get_lammps_calculator(self.potential_path, self.pot_config, self.work_dir)
