import logging
import shutil
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator
from ase.phonons import Phonons

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class PhononCalc:
    """
    Calculates phonon stability using finite displacements.
    """

    def __init__(self, supercell_matrix: list[int] | tuple[int, int, int] = (2, 2, 2), displacement: float = 0.01) -> None:
        self.supercell_matrix = tuple(supercell_matrix)
        self.displacement = displacement

    def calculate(
        self, structure: Structure, calculator: Calculator, workdir: Path
    ) -> tuple[bool, Structure | None]:
        """
        Check phonon stability.

        Args:
            structure: The unit cell structure.
            calculator: ASE calculator with the potential attached.
            workdir: Directory to store phonon calculation files.

        Returns:
            (is_stable, failed_structure)
        """
        workdir.mkdir(parents=True, exist_ok=True)
        atoms = structure.to_ase()
        atoms.calc = calculator

        # Prepare Phonons object
        phonon_dir = workdir / "phonons"
        if phonon_dir.exists():
            shutil.rmtree(phonon_dir)
        phonon_dir.mkdir()

        name = str(phonon_dir / "phonon")

        try:
            ph = Phonons(atoms, calculator, supercell=self.supercell_matrix, delta=self.displacement, name=name)
            ph.run()

            # Read forces and calculate band structure
            ph.read(acoustic=True)

            try:
                # auto-detect path
                path = atoms.cell.bandpath(npoints=50) # type: ignore[no-untyped-call]
                bs = ph.get_band_structure(path)
                frequencies = bs.energies
            except Exception as e:
                logger.warning(f"Could not determine bandpath: {e}. Checking Gamma point only.")
                frequencies = ph.get_frequencies(q=[0, 0, 0])

            is_stable = True
            if np.iscomplexobj(frequencies):
                 if np.any(np.abs(np.imag(frequencies)) > 1e-3):
                     is_stable = False

            if not is_stable:
                logger.info("Phonon instability detected.")
                supercell_atoms = ph.atoms
                failed_struct = Structure.from_ase(supercell_atoms)
                failed_struct.tags["provenance"] = "phonon_instability"
                return False, failed_struct

            return True, None

        except Exception:
            logger.exception("Phonon calculation failed")
            # If calculation fails, we flag as unstable
            return False, structure
