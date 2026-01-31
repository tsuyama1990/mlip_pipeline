from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.eos import EquationOfState

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.base import BaseValidator


class EOSValidator(BaseValidator):
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
        volumes = []
        energies = []

        # Generate scaled structures
        n_points = self.config.eos_n_points
        vol_range = self.config.eos_vol_range
        scales = np.linspace(1 - vol_range, 1 + vol_range, n_points)

        original_cell = self.structure.get_cell()

        for scale in scales:
            atoms = self.structure.copy()  # type: ignore[no-untyped-call]
            # Scale volume by scale factor (scale lattice constant by cbrt(scale))
            # Wait, linear scale vs volume scale.
            # SPEC: "volumes from 0.9 V0 to 1.1 V0"
            # So scale is volume scale.
            # Lattice constant scale factor is scale**(1/3)

            lat_scale = scale ** (1 / 3)
            atoms.set_cell(original_cell * lat_scale, scale_atoms=True)
            atoms.calc = self.calculator

            volumes.append(atoms.get_volume())
            energies.append(atoms.get_potential_energy())

        try:
            # Fit EOS
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, B = eos.fit()
            # B is in eV/Angstrom^3, convert to GPa
            B_GPa = B * 160.21766208

            passed = B_GPa > 0

            # Generate Plot
            plot_path = self.work_dir / "eos_plot.png"
            self._plot_eos(eos, plot_path)

            metric = ValidationMetric(
                name="Bulk Modulus (EOS)", value=float(B_GPa), passed=passed
            )

            return ValidationResult(
                potential_id=self.potential_id,
                metrics=[metric],
                plots={"EOS": plot_path},
                overall_status="PASS" if passed else "FAIL",
            )
        except Exception as e:
            metric = ValidationMetric(
                name="Bulk Modulus (EOS)",
                value=0.0,
                passed=False,
                error_message=str(e),
            )
            return ValidationResult(
                potential_id=self.potential_id,
                metrics=[metric],
                overall_status="FAIL",
            )

    def _plot_eos(self, eos: EquationOfState, path: Path):
        try:
            # We use our own plotting to avoid GUI backend issues and customize
            fig, ax = plt.subplots()
            eos.plot(ax=ax, show=False)
            fig.savefig(path)
            plt.close(fig)
        except Exception:
            # Fallback if ASE plot fails or matplotlib issues
            pass
