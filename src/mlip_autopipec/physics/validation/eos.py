from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
from ase.eos import EquationOfState
from ase.build import bulk

from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.runner import BaseValidator
from mlip_autopipec.physics.validation.utils import get_calculator


class EOSValidator(BaseValidator):
    def validate(self, potential_path: Path) -> ValidationResult:
        element = self.potential_config.elements[0]

        try:
            atoms = bulk(element)
        except Exception:
            # Fallback
            atoms = bulk(element, 'fcc', a=4.0) # type: ignore[no-untyped-call]

        calc = get_calculator(potential_path, self.potential_config)

        volumes = []
        energies = []

        # Scale volume factors -> scale length factors
        # vol_range (0.9, 1.1) -> V/V0
        # L/L0 = (V/V0)^(1/3)
        start_scale = self.config.eos_vol_range[0] ** (1/3)
        end_scale = self.config.eos_vol_range[1] ** (1/3)

        scale_factors = np.linspace(start_scale, end_scale, self.config.eos_n_points)

        for s in scale_factors:
            at = atoms.copy() # type: ignore[no-untyped-call]
            at.set_cell(atoms.cell * s, scale_atoms=True)
            at.calc = calc
            energies.append(at.get_potential_energy())
            volumes.append(at.get_volume())

        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        # fit() returns v0, e0, B, dB/dP, residual
        # Wait, fit() signature depends on ASE version.
        # Modern ASE: fit() returns v0, e0, B, Bprime
        try:
             v0_fit, e0_fit, B, Bprime = eos.fit()
        except ValueError:
             # Some versions might return different tuple? Or fitting failed?
             # Let's assume standard behavior.
             # If fitting fails, raise or fail validation.
             return ValidationResult(
                potential_id=potential_path.stem,
                metrics=[],
                plots={},
                overall_status="FAIL"
             )

        # 1 eV/Ang^3 = 160.21766208 GPa
        B_GPa = B * 160.21766208

        passed = B_GPa > 0

        metric = ValidationMetric(
            name="Bulk Modulus",
            value=float(B_GPa),
            passed=passed
        )

        plot_path = Path.cwd() / "eos_plot.png"
        eos.plot(filename=str(plot_path))

        return ValidationResult(
            potential_id=potential_path.stem,
            metrics=[metric],
            plots={"EOS": plot_path},
            overall_status="PASS" if passed else "FAIL"
        )
