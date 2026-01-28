import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS

from mlip_autopipec.config.schemas.validation import ElasticConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.validation.utils import load_calculator

logger = logging.getLogger(__name__)


class ElasticityValidator:
    """
    Validates Elastic Constants and Mechanical Stability.
    """

    def __init__(self, config: ElasticConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, atoms: Atoms, potential_path: Path) -> ValidationResult:
        logger.info("Starting Elasticity Validation...")

        if not self.config.enabled:
            return ValidationResult(module="elastic", passed=True, metrics=[], error=None)

        try:
            calc = load_calculator(potential_path)

            # Working copy
            atoms = atoms.copy()  # type: ignore
            atoms.calc = calc

            # 1. Relax structure first (Cell + Positions)
            # Elastic constants require equilibrium
            logger.info("Relaxing structure for elasticity check...")
            ucf = UnitCellFilter(atoms)
            opt = LBFGS(ucf, logfile=str(self.work_dir / "relax.log"))
            opt.run(fmax=0.01) # Strict relaxation

            # 2. Calculate Elastic Constants
            C_matrix = self._calculate_stiffness_matrix(atoms)

            # 3. Check Stability (Positive Definite)
            eigvals = np.linalg.eigvalsh(C_matrix)
            min_eig = np.min(eigvals)
            passed = min_eig > 0.0

            # Formatting metrics
            # We report Voigt notation C11, C12, C44 for cubic, or just matrix stats
            # For simplicity, we report min eigenvalue and full matrix in details

            details = {
                "C_matrix_GPa": C_matrix.tolist(),
                "eigenvalues": eigvals.tolist(),
                "min_eigenvalue": float(min_eig),
                "status": "Stable" if passed else "Unstable (Non-positive definite)"
            }

            metric = ValidationMetric(
                name="elastic_stability",
                value=float(min_eig),
                unit="GPa",
                passed=passed,
                details=details
            )

            return ValidationResult(
                module="elastic",
                passed=passed,
                metrics=[metric],
                error=None
            )

        except Exception as e:
            logger.exception("Elasticity validation failed")
            return ValidationResult(
                module="elastic",
                passed=False,
                error=str(e)
            )

    def _calculate_stiffness_matrix(self, atoms: Atoms) -> np.ndarray:
        """
        Calculates 6x6 Stiffness Matrix (Voigt notation) in GPa.
        Uses finite difference of stress-strain.
        """
        base_cell = atoms.get_cell()

        # 6 Voigt strains
        # 0: xx, 1: yy, 2: zz, 3: yz, 4: xz, 5: xy

        strains = np.linspace(
            -self.config.max_distortion,
            self.config.max_distortion,
            self.config.num_points
        )

        C = np.zeros((6, 6))

        # We need to compute stress for each strain pattern
        # Stress = C * Strain
        # We apply one strain component at a time and measure all stress components.
        # sigma_i = C_ij * epsilon_j

        # Map Voigt index to matrix indices
        voigt_map = {
            0: (0, 0), 1: (1, 1), 2: (2, 2),
            3: (1, 2), 4: (0, 2), 5: (0, 1)
        }

        for j in range(6):
            # Apply strain j
            stresses = [] # Stores (stress_voigt, strain_value)

            for eps in strains:
                if abs(eps) < 1e-8:
                    continue # Skip zero strain to avoid noise, or include it?
                    # Usually better to include if we fit line.

                strain_tensor = np.zeros((3, 3))
                v_idx = voigt_map[j]

                strain_tensor[v_idx] = eps
                if j >= 3: # Shear needs symmetry
                     strain_tensor[v_idx[::-1]] = eps

                # Deform
                deformed = atoms.copy() # type: ignore
                deformed.set_cell(base_cell @ (np.eye(3) + strain_tensor), scale_atoms=True)
                deformed.calc = atoms.calc

                # Get stress (Voigt notation usually returned by ASE: xx, yy, zz, yz, xz, xy)
                # Unit: eV/A^3. Convert to GPa.
                stress = deformed.get_stress(voigt=True)
                # ASE returns: [xx, yy, zz, yz, xz, xy]
                # Note: ASE stress sign convention: usually compressive is negative?
                # Wait, ASE: "Positive stress implies tension".
                # Standard: sigma = C * epsilon.

                # Convert to GPa
                stress_GPa = stress / 1e-21 * 1.60218e-19 / 1e9 # eV/A^3 to GPa
                # 1 eV = 1.60218e-19 J
                # 1 A = 1e-10 m -> A^3 = 1e-30 m^3
                # factor = 1.60218e-19 / 1e-30 = 1.60218e11 Pa = 160.218 GPa
                factor = 160.21766208
                stress_GPa = stress * factor

                stresses.append(stress_GPa)

            # Fit line for each stress component i vs strain j
            # sigma_i = C_ij * eps_j + sigma_0
            stresses_arr = np.array(stresses) # Shape: (N_strains, 6)

            # Polyfit degree 1
            # x = strains (excluding 0 if we skipped, but here we kept them in loop logic match)
            # Actually I filtered 0? No, I skipped it.
            # I should filter strains list too.
            valid_strains = [s for s in strains if abs(s) >= 1e-8]

            if len(valid_strains) < 2:
                # Should not happen with validation defaults
                continue

            for i in range(6):
                # Fit sigma_i vs strain
                slope, intercept = np.polyfit(valid_strains, stresses_arr[:, i], 1)
                C[i, j] = slope

        # Symmetrize C?
        # C_ij should be symmetric.
        C = (C + C.T) / 2.0

        return C
