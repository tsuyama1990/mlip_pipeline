from pathlib import Path
import numpy as np
from ase.eos import EquationOfState

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric
from mlip_autopipec.physics.validation.utils import get_lammps_calculator

class EOSValidator:
    """
    Validates the Equation of State (EOS) by fitting Birch-Murnaghan equation.
    Checks if the Bulk Modulus is positive.
    """
    def __init__(self, val_config: ValidationConfig, pot_config: PotentialConfig, potential_path: Path, work_dir: Path = Path("_work_validation/eos")):
        self.val_config = val_config
        self.pot_config = pot_config
        self.potential_path = potential_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, structure: Structure) -> ValidationMetric:
        atoms = structure.to_ase()

        range_frac = self.val_config.eos_vol_range
        n_points = self.val_config.eos_n_points

        # Volume scaling factors
        scales = np.linspace(1.0 - range_frac, 1.0 + range_frac, n_points)
        volumes = []
        energies = []

        calc = self._get_calculator()

        original_cell = atoms.get_cell()

        for scale in scales:
            # Create scaled copy
            atoms_scaled = atoms.copy() # type: ignore[no-untyped-call]
            l_scale = scale**(1.0/3.0)
            atoms_scaled.set_cell(original_cell * l_scale, scale_atoms=True) # type: ignore[no-untyped-call]

            atoms_scaled.calc = calc
            try:
                e = atoms_scaled.get_potential_energy()
                volumes.append(atoms_scaled.get_volume())
                energies.append(e)
            except Exception as e:
                return ValidationMetric(
                    name="Bulk Modulus (EOS)",
                    value=0.0,
                    passed=False,
                    message=f"Calculation failed: {e}"
                )

        # Fit Birch-Murnaghan
        try:
            eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
            v0_fit, e0_fit, B = eos.fit()

            # B is in eV/Ang^3. Convert to GPa.
            # 1 eV/Ang^3 = 160.21766208 GPa
            B_GPa = B * 160.21766208

            # Simple stability check: Bulk modulus must be positive
            passed = B_GPa > 0

            return ValidationMetric(
                name="Bulk Modulus (EOS)",
                value=B_GPa,
                passed=passed,
                message=f"B = {B_GPa:.2f} GPa"
            )
        except Exception as e:
             return ValidationMetric(
                    name="Bulk Modulus (EOS)",
                    value=0.0,
                    passed=False,
                    message=f"Fitting failed: {e}"
                )

    def _get_calculator(self):
        return get_lammps_calculator(self.potential_path, self.pot_config, self.work_dir)
