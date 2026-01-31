from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric
from mlip_autopipec.physics.validation.utils import get_lammps_calculator

class PhononValidator:
    def __init__(self, val_config: ValidationConfig, pot_config: PotentialConfig, potential_path: Path, work_dir: Path = Path("_work_validation/phonon")):
        self.val_config = val_config
        self.pot_config = pot_config
        self.potential_path = potential_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, structure: Structure) -> Tuple[ValidationMetric, Optional[Path]]:
        try:
            # 1. Setup Phonopy
            phonopy_obj = self._setup_phonopy(structure)

            # 2. Calculate Forces
            self._calculate_forces(phonopy_obj)

            # 3. Produce Force Constants
            phonopy_obj.produce_force_constants()

            # 4. Check Stability (Mesh)
            frequencies = self._get_frequencies(phonopy_obj)
            min_freq = float(np.min(frequencies))

            passed = min_freq >= self.val_config.phonon_tolerance

            # 5. Plot (Band Structure)
            plot_path = self._plot_band_structure(phonopy_obj)

            metric = ValidationMetric(
                name="Phonon Stability",
                value=min_freq,
                passed=passed,
                message=f"Min Freq: {min_freq:.4f} THz. " + ("Stable" if passed else "Unstable")
            )
            return metric, plot_path

        except Exception as e:
            return ValidationMetric(
                name="Phonon Stability", value=-999.0, passed=False, message=str(e)
            ), None

    def _setup_phonopy(self, structure: Structure) -> Phonopy:
        atoms = structure.to_ase()
        # Fix: PhonopyAtoms constructor args vs attributes
        # PhonopyAtoms(symbols=..., positions=..., cell=...)
        # We should iterate correctly.
        # However, checking mypy error: "PhonopyAtoms" has no attribute "get_chemical_symbols"
        # Correct, PhonopyAtoms uses .symbols, .positions, .cell attributes.
        # But we are constructing it here, not reading.
        # Wait, the error was in _calculate_forces where we READ from PhonopyAtoms.

        unitcell = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions()
        )

        supercell_matrix = list(self.val_config.phonon_supercell)
        phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
        phonon.generate_displacements(distance=0.01)
        return phonon

    def _calculate_forces(self, phonon: Phonopy) -> None:
        supercells = phonon.supercells_with_displacements
        calc = self._get_calculator()

        forces_set = []
        if supercells is None:
            return

        for p_atoms in supercells:
            if p_atoms is None:
                continue

            # Fix: PhonopyAtoms attributes access
            # p_atoms is a PhonopyAtoms object.
            # It has .symbols, .cell, .scaled_positions (properties or attributes)

            ase_atoms = Atoms(
                symbols=p_atoms.symbols,
                cell=p_atoms.cell,
                scaled_positions=p_atoms.scaled_positions,
                pbc=True
            )
            ase_atoms.calc = calc
            forces = ase_atoms.get_forces()
            forces_set.append(forces)

        phonon.forces = forces_set

    def _get_frequencies(self, phonon: Phonopy) -> np.ndarray:
        mesh = [20, 20, 20]
        phonon.run_mesh(mesh)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict['frequencies']
        return frequencies.flatten()

    def _plot_band_structure(self, phonon: Phonopy) -> Path:
        phonon.auto_band_structure(plot=True)
        # removed unused bs assignment

        bs_plot = phonon.plot_band_structure()

        output_path = self.work_dir / "phonon_dispersion.png"
        bs_plot.savefig(output_path)
        plt.close(bs_plot)
        return output_path

    def _get_calculator(self):
        return get_lammps_calculator(self.potential_path, self.pot_config, self.work_dir)
