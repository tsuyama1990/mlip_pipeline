from typing import Optional
from pathlib import Path
from ase import Atoms
from ase.eos import EquationOfState
import numpy as np

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.utils import get_lammps_calculator

class EOSValidator:
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

    def validate(self, reference_structure: Atoms) -> ValidationResult:
        # 1. Setup
        work_dir = Path("validation_work/eos")
        work_dir.mkdir(parents=True, exist_ok=True)

        calc = self._get_calculator(work_dir)

        # 2. Generate structures
        # V0 +/- range
        n_points = self.config.eos_n_points
        vol_range = self.config.eos_vol_range
        scales = np.linspace(1.0 - vol_range, 1.0 + vol_range, n_points)

        volumes = []
        energies = []

        # Ensure we don't modify the reference in place
        atoms = reference_structure.copy() # type: ignore[no-untyped-call]
        atoms.calc = calc

        cell_0 = atoms.get_cell()

        for scale in scales:
            # Scale volume by factor 'scale' -> linear dim by scale^(1/3)
            # Actually scale is Vol/Vol0 usually in these contexts, but let's assume it is.
            # ase.eos usually wants V, E.
            # We scale the cell.
            linear_scale = scale**(1/3.0)
            atoms.set_cell(cell_0 * linear_scale, scale_atoms=True) # type: ignore[no-untyped-call]

            try:
                energy = atoms.get_potential_energy()
                volumes.append(atoms.get_volume())
                energies.append(energy)
            except Exception as e:
                # Calculation failed
                pass

        # 3. Fit EOS
        passed = False
        bulk_modulus = 0.0
        error_msg = None
        plot_path = work_dir / "eos_plot.png"

        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, B = eos.fit()
            bulk_modulus = B / 1.602176634e-19 * 1.0e-21 # eV/A^3 to GPa
            # ASE units: B is in eV/A^3.
            # 1 eV = 1.602e-19 J. 1 A^3 = 1e-30 m^3.
            # eV/A^3 = 1.602e-19 / 1e-30 = 1.602e11 Pa = 160.2 GPa.
            bulk_modulus = B * 160.21766208

            passed = bulk_modulus > 0

            # Plot
            try:
                import matplotlib.pyplot as plt
                eos.plot(str(plot_path))
                plt.close() # Ensure we close the figure
            except ImportError:
                pass
            except Exception:
                pass

        except Exception as e:
            error_msg = str(e)
            passed = False

        metric = ValidationMetric(
            name="Bulk Modulus",
            value=float(bulk_modulus),
            passed=passed,
            message=error_msg
        )

        status = "PASS" if passed else "FAIL"

        return ValidationResult(
            potential_id=self.potential_path.stem,
            metrics=[metric],
            plots={"eos": plot_path} if plot_path.exists() else {},
            overall_status=status
        )
