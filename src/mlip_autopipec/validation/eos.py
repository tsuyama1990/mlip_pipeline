import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.eos import EquationOfState
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS

from mlip_autopipec.config.schemas.validation import EOSConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.validation.utils import load_calculator

logger = logging.getLogger(__name__)


class EOSValidator:
    """
    Validates Equation of State (Bulk Modulus, Equilibrium Volume).
    """

    def __init__(self, config: EOSConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, atoms: Atoms, potential_path: Path) -> ValidationResult:
        logger.info("Starting EOS Validation...")

        if not self.config.enabled:
            return ValidationResult(module="eos", passed=True, metrics=[], error=None)

        try:
            calc = load_calculator(potential_path)

            # Working copy
            atoms = atoms.copy()  # type: ignore
            atoms.calc = calc

            # 1. Relax first
            logger.info("Relaxing structure for EOS check...")
            ucf = UnitCellFilter(atoms)
            opt = LBFGS(ucf, logfile=str(self.work_dir / "relax.log"))  # type: ignore[arg-type]
            opt.run(fmax=0.01)

            # 2. Generate Scaling factors
            volumes = []
            energies = []

            base_cell = atoms.get_cell()

            # strains are volumetric strains e.g. -0.1, 0.0, 0.1
            strains = np.linspace(-self.config.strain_max, self.config.strain_max, self.config.num_points)

            for strain in strains:
                vol_scale = 1.0 + strain
                linear_scale = vol_scale ** (1.0 / 3.0)

                scaled_atoms = atoms.copy() # type: ignore
                scaled_atoms.set_cell(base_cell * linear_scale, scale_atoms=True)
                scaled_atoms.calc = calc

                v = scaled_atoms.get_volume()
                e = scaled_atoms.get_potential_energy()

                volumes.append(v)
                energies.append(e)

            # 3. Fit EOS
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, B = eos.fit()

            # Save plot
            eos.plot(str(self.work_dir / "eos.png"))

            # 4. Check Convexity / Stability
            # B (Bulk Modulus) should be positive.
            # Convert B to GPa? ase.eos B is in eV/A^3.
            # 1 eV/A^3 = 160.2 GPa.
            B_GPa = B * 160.21766208

            passed = B_GPa > 0.0

            details = {
                "v0": float(v0),
                "e0": float(e0),
                "B_GPa": float(B_GPa),
                "status": "Stable" if passed else "Unstable (B <= 0)"
            }

            metric = ValidationMetric(
                name="bulk_modulus",
                value=float(B_GPa),
                unit="GPa",
                passed=passed,
                details=details
            )

            return ValidationResult(
                module="eos",
                passed=passed,
                metrics=[metric],
                error=None
            )

        except Exception as e:
            logger.exception("EOS validation failed")
            return ValidationResult(
                module="eos",
                passed=False,
                error=str(e)
            )
