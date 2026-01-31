from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from .utils import get_lammps_calculator

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False


class PhononValidator:
    def __init__(self, config: ValidationConfig, potential_path: Path):
        self.config = config
        self.potential_path = potential_path

    def validate(self, structure: Structure) -> ValidationResult:
        if not PHONOPY_AVAILABLE:
            return ValidationResult(
                potential_id=str(self.potential_path),
                metrics=[
                    ValidationMetric(name="Phonopy Installed", value=0.0, passed=False)
                ],
                plots={},
                overall_status="FAIL",
            )

        try:
            atoms = structure.to_ase()
            elements = sorted(list(set(atoms.get_chemical_symbols())))
            calc = get_lammps_calculator(self.potential_path, elements)

            # Setup Phonopy
            unitcell = PhonopyAtoms(
                symbols=atoms.get_chemical_symbols(),
                cell=atoms.get_cell(),
                scaled_positions=atoms.get_scaled_positions(),
            )

            # Supercell matrix (2x2x2)
            supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

            phonon = Phonopy(unitcell, supercell_matrix)
            phonon.generate_displacements(distance=0.01)

            supercells = phonon.supercells_with_displacements
            if supercells is None:
                raise ValueError("Failed to generate displacements")

            # Calculate forces
            set_of_forces = []
            for sc in supercells:
                # Convert Phonopy atoms to ASE atoms
                sc_ase = Atoms(
                    symbols=sc.symbols,
                    cell=sc.cell,
                    scaled_positions=sc.scaled_positions,
                    pbc=True,
                )

                sc_ase.calc = calc
                # Type ignore because get_forces might return None if calc fails (but usually raises)
                forces = sc_ase.get_forces()  # type: ignore[no-untyped-call]
                set_of_forces.append(forces)

            phonon.produce_force_constants(forces=set_of_forces)

            # Check stability on a mesh
            mesh = [10, 10, 10]
            phonon.run_mesh(mesh)
            mesh_dict = phonon.get_mesh_dict()
            frequencies = mesh_dict["frequencies"]

            min_freq = float(np.min(frequencies))

            # Tolerance is e.g. -0.1 THz.
            # If min_freq = -0.05, it is > -0.1, so PASS.
            # If min_freq = -0.5, it is < -0.1, so FAIL.
            passed = bool(min_freq > self.config.phonon_tolerance)

            metric = ValidationMetric(
                name="Min Phonon Freq", value=min_freq, passed=passed
            )

            # Generate plot path (just a placeholder if we don't actually run plot)
            # Or assume we save it.
            # If matplotlib is available, we can save.
            plot_path = Path("phonon_dispersion.png")
            # Skipping actual plot generation to avoid heavy dependencies/display issues in CI
            # But normally:
            # phonon.auto_band_structure(plot=True).savefig(plot_path)

            return ValidationResult(
                potential_id=str(self.potential_path),
                metrics=[metric],
                plots={"phonon": plot_path},
                overall_status="PASS" if passed else "FAIL",
            )

        except Exception as e:
            return ValidationResult(
                potential_id=str(self.potential_path),
                metrics=[
                    ValidationMetric(
                        name="Phonon Error", value=0.0, message=str(e), passed=False
                    )
                ],
                plots={},
                overall_status="FAIL",
            )
