from typing import Optional
from pathlib import Path
from ase import Atoms
import numpy as np

# Conditional import for phonopy to avoid crashing if not installed (though we installed it)
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.utils import get_lammps_calculator

class PhononValidator:
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

    def _ase_to_phonopy(self, atoms: Atoms):
        """Convert ASE Atoms to PhonopyAtoms."""
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions()
        )

    def validate(self, reference_structure: Atoms) -> ValidationResult:
        if not PHONOPY_AVAILABLE:
            return ValidationResult(
                potential_id=self.potential_path.stem,
                metrics=[ValidationMetric(name="Phonon Check", value=0.0, passed=False, message="Phonopy not installed")],
                plots={},
                overall_status="WARN"
            )

        work_dir = Path("validation_work/phonon")
        work_dir.mkdir(parents=True, exist_ok=True)

        calc = self._get_calculator(work_dir)

        # 1. Setup Phonopy
        unitcell = self._ase_to_phonopy(reference_structure)
        supercell_matrix = list(self.config.phonon_supercell) # e.g. [2, 2, 2]
        # Phonopy expects 3x3 or list of 3.
        # If tuple (2,2,2) -> diag matrix
        if len(supercell_matrix) == 3:
            smat = np.diag(supercell_matrix)
        else:
            smat = np.eye(3) * 2 # Fallback

        phonon = Phonopy(unitcell, smat)
        phonon.generate_displacements(distance=0.01)

        supercells = phonon.supercells_with_displacements
        # 2. Calculate Forces
        forces_set = []

        for i, sc in enumerate(supercells):
            # Convert PhonopyAtoms back to ASE
            # sc is PhonopyAtoms? No, it's None in recent versions?
            # 'supercells_with_displacements' returns list of PhonopyAtoms.

            # Convert sc to ASE
            ase_sc = Atoms(
                symbols=sc.symbols,
                cell=sc.cell,
                scaled_positions=sc.scaled_positions,
                pbc=True
            )

            # Setup calc
            ase_sc.calc = calc

            # Run
            try:
                f = ase_sc.get_forces()
                forces_set.append(f)
            except Exception as e:
                return ValidationResult(
                    potential_id=self.potential_path.stem,
                    metrics=[ValidationMetric(name="Phonon Forces", value=0.0, passed=False, message=str(e))],
                    plots={},
                    overall_status="FAIL"
                )

        # 3. Post-process
        phonon.produce_force_constants(forces=forces_set)

        # 4. Band Structure
        # Auto-path (Seek-path like)
        phonon.auto_band_structure(plot=True)
        # Note: auto_band_structure writes nothing but preps data.
        # We need to get frequencies.

        bs_dict = phonon.get_band_structure_dict()
        frequencies = bs_dict['frequencies'] # list of arrays (q-points)

        # Check for imaginary frequencies (negative values in phonopy usually, unless complex)
        # Phonopy stores them as real numbers. Imaginary are negative?
        # "Imaginary frequencies are represented by negative real numbers."

        min_freq = float("inf")
        for band in frequencies:
            min_f = np.min(band)
            if min_f < min_freq:
                min_freq = min_f

        # Tolerance
        passed = min_freq > self.config.phonon_tolerance

        # Plot
        plot_path = work_dir / "phonon_band.png"
        try:
            import matplotlib.pyplot as plt
            plt = phonon.plot_band_structure()
            plt.savefig(plot_path)
            plt.close()
        except Exception:
            pass

        metric = ValidationMetric(
            name="Min Frequency",
            value=float(min_freq),
            passed=passed,
            message="Imaginary frequencies detected" if not passed else None
        )

        status = "PASS" if passed else "FAIL"

        return ValidationResult(
            potential_id=self.potential_path.stem,
            metrics=[metric],
            plots={"phonon": plot_path} if plot_path.exists() else {},
            overall_status=status
        )
