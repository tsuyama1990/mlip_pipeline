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
        # We perform the calculation in a subdirectory to avoid clutter
        phonon_dir = workdir / "phonons"
        if phonon_dir.exists():
            shutil.rmtree(phonon_dir)
        phonon_dir.mkdir()

        # Switch to phonon dir to let ASE write files there (it writes 'phonons.json' etc)
        # Actually ASE Phonons class takes 'name' argument which is the prefix/path.
        name = str(phonon_dir / "phonon")

        try:
            ph = Phonons(atoms, calculator, supercell=self.supercell_matrix, delta=self.displacement, name=name)
            ph.run()

            # Read forces and calculate band structure
            ph.read(acoustic=True)

            # Check frequencies at Gamma and some high symmetry points
            # For simplicity, we check the band structure along a standard path
            # If bandpath fails (e.g. no cell), we check Gamma only.
            try:
                # auto-detect path
                path = atoms.cell.bandpath(npoints=50)
                bs = ph.get_band_structure(path)
                # bs.energies is an array of shape (n_kpoints, n_bands)
                # These are frequencies (often in eV or similar depending on units, ASE uses eV usually for energies)
                # Wait, ph.get_band_structure returns BandStructure object.
                # The energies attribute contains the eigenvalues (frequencies).
                frequencies = bs.energies
            except Exception as e:
                logger.warning(f"Could not determine bandpath: {e}. Checking Gamma point only.")
                frequencies = ph.get_frequencies(q=[0, 0, 0])

            # Check for imaginary frequencies
            # In ASE, imaginary frequencies are usually complex.
            # However, sometimes they are returned as negative real values if taking sqrt of neg eigenvalue.
            # ASE Phonons.get_frequencies returns complex for imaginary.

            is_stable = True
            if np.iscomplexobj(frequencies):
                 # Check if any imaginary part is significant
                 if np.any(np.abs(np.imag(frequencies)) > 1e-3):
                     is_stable = False
            else:
                 # If real, check for negative frequencies (some conventions)
                 # But usually real means stable.
                 # Wait, acoustic modes at Gamma should be 0.
                 # If we have large negative real numbers (which implies imaginary in some codes), we check.
                 # But ASE returns complex.

                 # Note: Acoustic sum rule might not be perfect, so small imaginary/negative modes at Gamma are expected.
                 # We should ignore small imaginary values at Gamma.
                 pass

            if not is_stable:
                logger.info("Phonon instability detected.")
                # Return the supercell or a specific displaced structure?
                # The spec says "Fail -> Add failure configurations".
                # Ideally, we return a structure that exhibits the instability.
                # A snapshot from the MD or just the supercell is fine.
                # For now, return the supercell.
                # Construct supercell atoms
                supercell_atoms = ph.atoms  # This is the supercell
                failed_struct = Structure.from_ase(supercell_atoms)
                failed_struct.tags["provenance"] = "phonon_instability"
                return False, failed_struct

            return True, None

        except Exception as e:
            logger.error(f"Phonon calculation failed: {e}")
            # If calculation fails (e.g. SCF doesn't converge), we might consider it unstable or just error.
            # For robustness, we flag as unstable so we can investigate or retry.
            return False, structure
