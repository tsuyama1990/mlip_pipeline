from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.base import BaseValidator

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False


class PhononValidator(BaseValidator):
    def __init__(
        self,
        structure: Atoms,
        calculator: Calculator,
        config: ValidationConfig,
        work_dir: Path,
        potential_id: str,
    ):
        if not PHONOPY_AVAILABLE:
            raise RuntimeError("Phonopy not installed.")

        self.structure = structure.copy()  # type: ignore[no-untyped-call]
        self.calculator = calculator
        self.config = config
        self.work_dir = work_dir
        self.potential_id = potential_id
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> ValidationResult:
        try:
            # 1. Setup Phonopy
            unitcell = self._ase_to_phonopy(self.structure)
            # Use configurable supercell
            sc = self.config.phonon_supercell
            supercell_matrix = [[sc[0], 0, 0], [0, sc[1], 0], [0, 0, sc[2]]]

            phonon = Phonopy(unitcell, supercell_matrix)

            # 2. Generate displacements
            phonon.generate_displacements(distance=0.01)
            supercells = phonon.supercells_with_displacements

            # 3. Calculate forces
            forces_set = []
            if supercells is not None:
                for sc in supercells:
                    # Convert Phonopy atom to ASE
                    ase_atoms = self._phonopy_to_ase(sc)
                    ase_atoms.calc = self.calculator
                    forces = ase_atoms.get_forces()
                    forces_set.append(forces)

            # 4. Set forces
            phonon.set_forces(forces_set)  # type: ignore[attr-defined]
            phonon.produce_force_constants()

            # 5. Check stability on Mesh (faster/more comprehensive than band structure for stability)
            phonon.run_mesh([10, 10, 10])
            mesh_dict = phonon.get_mesh_dict()
            frequencies = mesh_dict["frequencies"]
            min_freq = np.min(frequencies)

            # Tolerance (from SPEC: < -0.1 THz is FAIL)
            # Note: Imaginary freqs are returned as negative real numbers in Phonopy usually,
            # or we need to check if they are complex?
            # Phonopy returns imaginary frequencies as negative values (convention).

            passed = min_freq > self.config.phonon_tolerance

            # 6. Plot Band Structure
            plot_path = self.work_dir / "phonon_band_structure.png"
            self._plot_band_structure(phonon, plot_path)

            metric = ValidationMetric(
                name="Phonon Stability (Min Freq)", value=float(min_freq), passed=passed
            )

            return ValidationResult(
                potential_id=self.potential_id,
                metrics=[metric],
                plots={"Phonon Dispersion": plot_path},
                overall_status="PASS" if passed else "FAIL",
            )

        except Exception as e:
            return ValidationResult(
                potential_id=self.potential_id,
                metrics=[
                    ValidationMetric(
                        name="Phonon", value=0.0, passed=False, error_message=str(e)
                    )
                ],
                overall_status="FAIL",
            )

    def _ase_to_phonopy(self, atoms: Atoms) -> "PhonopyAtoms":
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            positions=atoms.get_positions(),
        )

    def _phonopy_to_ase(self, phonopy_atoms: "PhonopyAtoms") -> Atoms:
        return Atoms(
            symbols=phonopy_atoms.symbols,
            cell=phonopy_atoms.cell,
            positions=phonopy_atoms.positions,
            pbc=True,
        )

    def _plot_band_structure(self, phonon: "Phonopy", path: Path):
        try:
            # Auto band structure
            phonon.auto_band_structure(plot=True).savefig(path)
            # Note: auto_band_structure returns a plt object (module or figure?)
            # Usually it returns the module plt.
            plt.close()
        except Exception:
            pass
