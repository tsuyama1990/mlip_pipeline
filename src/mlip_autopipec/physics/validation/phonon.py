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
    """
    Validates phonon stability of a structure using Phonopy.
    Checks for imaginary frequencies in the band structure.
    """
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
            # We process forces sequentially.
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
        """
        Calculates forces for each supercell with displacement.
        """
        supercells = phonon.supercells_with_displacements
        if supercells is None:
            return

        calc = self._get_calculator()

        # Phonopy requires all forces to be set.
        # To avoid OOM with large systems, we want to avoid holding all ASE atoms in memory.
        # But we must hold all force arrays (N, 3) because Phonopy stores them.
        # The force arrays are much smaller than the Atoms objects.
        # We use a generator to process the calculation one by one.

        forces_list = []

        for i, p_atoms in enumerate(supercells):
            if p_atoms is None:
                continue

            # Create ASE atoms only for the calculation scope
            ase_atoms = Atoms(
                symbols=p_atoms.symbols,
                cell=p_atoms.cell,
                scaled_positions=p_atoms.scaled_positions,
                pbc=True
            )
            ase_atoms.calc = calc

            # Compute forces
            forces = ase_atoms.get_forces()

            # Store only the force array
            forces_list.append(forces)

            # ase_atoms is discarded here

        phonon.forces = forces_list

    def _get_frequencies(self, phonon: Phonopy) -> np.ndarray:
        mesh = [20, 20, 20]
        phonon.run_mesh(mesh)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict['frequencies']
        return frequencies.flatten()

    def _plot_band_structure(self, phonon: Phonopy) -> Path:
        phonon.auto_band_structure(plot=True)
        bs_plot = phonon.plot_band_structure()

        output_path = self.work_dir / "phonon_dispersion.png"
        bs_plot.savefig(output_path)
        plt.close(bs_plot)
        return output_path

    def _get_calculator(self):
        return get_lammps_calculator(self.potential_path, self.pot_config, self.work_dir)
