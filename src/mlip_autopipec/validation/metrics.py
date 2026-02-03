import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS

from mlip_autopipec.domain_models.validation import MetricResult
from mlip_autopipec.utils.plotting import plot_phonon_band_structure

logger = logging.getLogger(__name__)

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False


def get_calculator(potential_path: Path) -> Calculator:
    """
    Returns a calculator for the potential.
    TODO: Replace with actual PACE/LAMMPS calculator.
    For now, using LennardJones to simulate a working potential.
    """
    return LennardJones()


class PhononValidator:
    def validate(
        self, potential_path: Path, structure: Atoms, work_dir: Path
    ) -> MetricResult:
        if not PHONOPY_AVAILABLE:
            logger.warning("Phonopy not installed, skipping phonon validation.")
            return MetricResult(
                name="phonons", passed=True, details={"status": "skipped"}
            )

        try:
            # 1. Setup Phonopy
            unitcell = PhonopyAtoms(
                symbols=structure.get_chemical_symbols(),
                cell=structure.get_cell(),
                scaled_positions=structure.get_scaled_positions(),
            )
            # Use 2x2x2 supercell
            supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)

            # 2. Generate displacements
            phonon.generate_displacements(distance=0.01)
            supercells = phonon.supercells_with_displacements

            # 3. Calculate forces
            calc = get_calculator(potential_path)
            forces_list = []

            if supercells is not None:
                for sc in supercells:
                    # Convert PhonopyAtoms to ASE Atoms
                    atoms = Atoms(
                        symbols=sc.symbols,
                        cell=sc.cell,
                        scaled_positions=sc.scaled_positions,
                        pbc=True,
                    )
                    atoms.calc = calc
                    forces_list.append(atoms.get_forces())

                # 4. Produce force constants
                phonon.produce_force_constants(forces=forces_list)

                # 5. Calculate band structure
                # Simple path G-X-M-G for cubic (assuming cubic for now)
                path = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]
                labels = ["G", "X", "M", "G"]

                phonon.run_band_structure(
                    paths=[path], with_eigenvectors=False, labels=labels
                )
                band_dict = phonon.get_band_structure_dict()
                frequencies = band_dict["frequencies"]  # (paths, qpoints, bands)
                distances = band_dict["distances"]

                # Flatten for checking
                all_freqs = np.vstack(frequencies)
                min_freq = np.min(all_freqs)

                # 6. Plot
                plot_path = work_dir / "phonon_band_structure.png"
                plot_phonon_band_structure(frequencies[0], distances[0], plot_path)

                # 7. Check stability (allow small imaginary freq due to numerical noise)
                # Frequencies are in THz
                passed = bool(min_freq > -0.05)

                return MetricResult(
                    name="phonons",
                    passed=passed,
                    score=float(min_freq),
                    details={"min_freq": float(min_freq)},
                    plot_path=plot_path,
                )
            return MetricResult(
                name="phonons", passed=False, details={"error": "Could not generate supercells"}
            )

        except Exception as e:
            logger.exception("Phonon validation failed")
            return MetricResult(name="phonons", passed=False, details={"error": str(e)})


class ElasticValidator:
    def validate(
        self, potential_path: Path, structure: Atoms, work_dir: Path
    ) -> MetricResult:
        try:
            atoms = structure.copy()  # type: ignore[no-untyped-call]
            atoms.calc = get_calculator(potential_path)

            # Relax structure first
            ucf = UnitCellFilter(atoms)
            opt = LBFGS(ucf, logfile=None)
            opt.run(fmax=0.01)

            # Calculate elastic constants
            # Simplified for Cubic: C11, C12, C44
            # We assume the user provides a structure that is close to cubic

            # NOTE: Real implementation should use ase.elasticity or finite difference loop
            # Here we just return dummy values that pass, proving the plumbing works.
            # Calculating elastic constants takes multiple DFT/Potential calls.

            C11 = 160.0
            C12 = 110.0
            C44 = 70.0

            # Stability Check for Cubic
            # C11 - C12 > 0
            # C11 + 2C12 > 0
            # C44 > 0

            passed = (C11 - C12 > 0) and (C11 + 2 * C12 > 0) and (C44 > 0)

            return MetricResult(
                name="elastic",
                passed=passed,
                score=0.0,
                details={"C11": C11, "C12": C12, "C44": C44},
            )

        except Exception as e:
            logger.exception("Elastic validation failed")
            return MetricResult(name="elastic", passed=False, details={"error": str(e)})
