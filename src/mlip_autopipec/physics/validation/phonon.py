from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import (
    ValidationConfig,
    ValidationMetric,
)
from mlip_autopipec.physics.validation.utils import get_validation_calculator


class PhononValidator:
    def __init__(
        self,
        val_config: ValidationConfig,
        pot_config: PotentialConfig,
        work_dir: Path,
    ):
        self.val_config = val_config
        self.pot_config = pot_config
        self.work_dir = work_dir / "phonon"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(
        self, structure: Structure, potential_path: Path
    ) -> tuple[list[ValidationMetric], dict[str, Path]]:
        atoms_ase = structure.to_ase()

        # 1. Setup Phonopy
        # Convert ASE to PhonopyAtoms
        unitcell = PhonopyAtoms(
            symbols=atoms_ase.get_chemical_symbols(),  # type: ignore[no-untyped-call]
            cell=atoms_ase.get_cell(),  # type: ignore[no-untyped-call]
            positions=atoms_ase.get_positions(),  # type: ignore[no-untyped-call]
        )

        # Supercell size: 2x2x2 default
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

        phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)

        # 2. Generate displacements
        phonon.generate_displacements(distance=0.01)
        supercells = phonon.supercells_with_displacements

        if supercells is None:
            raise RuntimeError("Failed to generate displacements.")

        # 3. Calculate forces
        sets_of_forces = []
        calc = get_validation_calculator(potential_path, self.pot_config, self.work_dir)

        # We can try to reuse calculator if possible, or just re-attach
        for sc in supercells:
            if sc is None:
                continue
            # Convert PhonopyAtoms sc to ASE
            sc_ase = Atoms(
                symbols=sc.symbols,
                cell=sc.cell,
                positions=sc.positions,
                pbc=True,
            )
            sc_ase.calc = calc
            try:
                forces = sc_ase.get_forces()  # type: ignore[no-untyped-call]
            except Exception:
                # Calculation failed
                # Return fail metric
                return [
                    ValidationMetric(
                        name="Phonon Stability",
                        value=-999.9,
                        passed=False,
                        unit="THz",
                    )
                ], {}

            sets_of_forces.append(forces)

        # 4. Set forces and produce constants
        phonon.produce_force_constants(forces=sets_of_forces)

        # 5. Check Mesh for stability
        mesh = [8, 8, 8]
        phonon.run_mesh(mesh)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict["frequencies"]  # (q, band)

        min_freq = np.min(frequencies)

        # Tolerance check
        passed = min_freq > self.val_config.phonon_tolerance

        metric = ValidationMetric(
            name="Phonon Stability",
            value=float(min_freq),
            passed=bool(passed),
            unit="THz",
        )

        # 6. Plot Band Structure
        plot_path = self.work_dir / "phonon_band.png"
        plots = {}
        try:
            # Auto band structure
            bs = phonon.auto_band_structure(plot=True)
            bs.savefig(plot_path)
            plots["phonon_band_structure"] = plot_path
            # Close plot to avoid memory leak
            plt.close("all")
        except Exception:
            # Fallback if seekpath not installed or plotting fails
            pass

        return [metric], plots
