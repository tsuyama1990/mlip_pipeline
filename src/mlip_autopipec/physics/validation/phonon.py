from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import ase
from ase.build import bulk
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.validation.runner import BaseValidator
from mlip_autopipec.physics.validation.utils import get_calculator


class PhononValidator(BaseValidator):
    def validate(self, potential_path: Path) -> ValidationResult:
        element = self.potential_config.elements[0]
        try:
            atoms = bulk(element) # type: ignore[no-untyped-call]
        except Exception:
            atoms = bulk(element, 'fcc', a=4.0) # type: ignore[no-untyped-call]

        calc = get_calculator(potential_path, self.potential_config)

        # Phonopy setup
        unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                                cell=atoms.cell,
                                scaled_positions=atoms.get_scaled_positions())

        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)

        phonon.generate_displacements(distance=0.01)
        supercells = phonon.supercells_with_displacements

        set_of_forces = []
        # supercells is list of PhonopyAtoms (or None if not generated)
        if supercells is None:
             # Should not happen
             supercells = []

        for sc in supercells:
            # Convert PhonopyAtoms back to ASE
            ase_sc = ase.Atoms(symbols=sc.symbols,
                               cell=sc.cell,
                               scaled_positions=sc.scaled_positions,
                               pbc=True)
            ase_sc.calc = calc
            forces = ase_sc.get_forces()
            set_of_forces.append(forces)

        phonon.forces = set_of_forces # property setter
        phonon.produce_force_constants()

        # Simple path G-X-L-G for cubic
        path = [[[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0.5], [0, 0, 0]]]
        labels = ["G", "X", "L", "G"]

        phonon.run_band_structure(paths=path, labels=labels)  # type: ignore[arg-type]
        bs_dict = phonon.get_band_structure_dict()
        frequencies = bs_dict['frequencies'] # list of arrays

        min_freq = float("inf")
        for segment in frequencies:
            min_f = segment.min()
            if min_f < min_freq:
                min_freq = min_f

        passed = min_freq >= self.config.phonon_tolerance

        metric = ValidationMetric(
            name="Min Frequency",
            value=float(min_freq),
            passed=passed
        )

        plot_path = Path.cwd() / "phonon_plot.png"
        plt = phonon.plot_band_structure()
        plt.savefig(plot_path)
        plt.close() # Close to free memory

        return ValidationResult(
            potential_id=potential_path.stem,
            metrics=[metric],
            plots={"Phonon": plot_path},
            overall_status="PASS" if passed else "FAIL"
        )
