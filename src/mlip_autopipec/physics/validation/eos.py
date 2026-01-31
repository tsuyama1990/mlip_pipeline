from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.eos import EquationOfState
from ase.units import GPa

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationConfig, ValidationMetric
from mlip_autopipec.physics.validation.utils import get_validation_calculator


class EOSValidator:
    def __init__(
        self,
        val_config: ValidationConfig,
        pot_config: PotentialConfig,
        work_dir: Path,
    ):
        self.val_config = val_config
        self.pot_config = pot_config
        self.work_dir = work_dir / "eos"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(
        self, structure: Structure, potential_path: Path
    ) -> tuple[list[ValidationMetric], dict[str, Path]]:
        atoms = structure.to_ase()
        volumes = []
        energies = []

        # Deformations (Volume scaling)
        scalings = np.linspace(
            1.0 - self.val_config.eos_vol_range,
            1.0 + self.val_config.eos_vol_range,
            self.val_config.eos_n_points,
        )

        calc = get_validation_calculator(potential_path, self.pot_config, self.work_dir)
        atoms.calc = calc

        cell_0 = atoms.get_cell()

        for scale in scalings:
            # Isotropic scaling of volume -> scale^(1/3) for length
            l_scale = scale ** (1.0 / 3.0)
            atoms.set_cell(cell_0 * l_scale, scale_atoms=True)

            try:
                e = atoms.get_potential_energy()
                energies.append(e)
                volumes.append(atoms.get_volume())
            except Exception:
                # If calculation fails (e.g. lost atoms?), skip point
                pass

        if len(energies) < 3:
            return [
                ValidationMetric(
                    name="Bulk Modulus", value=0.0, passed=False, unit="GPa"
                )
            ], {}

        # Fit EOS
        # ASE EOS fit returns v0, e0, B
        # B is in eV/Angstrom^3
        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, B = eos.fit()

            # Convert B to GPa
            # In ASE: 1 eV/A^3 = 160.217 GPa (approx)
            # value / GPa -> value_in_GPa
            B_GPa = B / GPa

            passed = B_GPa > 0

            # Plot
            plot_path = self.work_dir / "eos.png"
            # ASE eos.plot() creates a figure if ax not provided, or uses ax.
            # We want to save it.
            # eos.plot() returns ax object usually.
            # We create figure first to control saving.
            fig, ax = plt.subplots()
            eos.plot(ax=ax, show=False)
            ax.set_title(f"EOS (B = {B_GPa:.1f} GPa)")
            fig.savefig(plot_path)
            plt.close(fig)

            metric = ValidationMetric(
                name="Bulk Modulus", value=float(B_GPa), unit="GPa", passed=passed
            )
            return [metric], {"eos_plot": plot_path}

        except Exception:
            # Fitting failed
            return [
                ValidationMetric(
                    name="Bulk Modulus", value=0.0, passed=False, unit="GPa"
                )
            ], {}
