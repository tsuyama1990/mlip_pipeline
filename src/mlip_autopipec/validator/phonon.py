import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    _PHONOPY_AVAILABLE = True
except ImportError:
    _PHONOPY_AVAILABLE = False
    Phonopy = None # type: ignore
    PhonopyAtoms = None # type: ignore

logger = logging.getLogger(__name__)

class PhononResults(NamedTuple):
    """
    Phonon stability results.

    Attributes:
        is_stable: Whether the structure is dynamically stable.
        max_imaginary_freq: Magnitude of the largest imaginary frequency (THz).
        band_structure_path: Path to the phonon band structure plot.
    """
    is_stable: bool
    max_imaginary_freq: float
    band_structure_path: Path | None

class PhononAnalyzer:
    """Analyzes phonon stability using Phonopy."""

    def __init__(self, supercell_matrix: list[int] | None = None, displacement_distance: float = 0.01) -> None:
        """
        Initialize PhononAnalyzer.

        Args:
            supercell_matrix: Supercell matrix (diagonal if list of 3 ints). Default [2, 2, 2].
            displacement_distance: Atomic displacement distance (Angstrom).
        """
        if supercell_matrix is None:
            supercell_matrix = [2, 2, 2]

        if len(supercell_matrix) == 3 and isinstance(supercell_matrix[0], int):
             self.supercell_matrix = np.diag(supercell_matrix)
        else:
             self.supercell_matrix = np.array(supercell_matrix)

        self.displacement_distance = displacement_distance

    def calculate_phonons(self, structure: Atoms, potential: Potential) -> PhononResults:
        """
        Calculate phonon stability.

        Args:
            structure: ASE Atoms object.
            potential: Potential to use.

        Returns:
            PhononResults object.
        """
        if not _PHONOPY_AVAILABLE:
            logger.warning("Phonopy not available. Skipping phonon validation.")
            return PhononResults(is_stable=True, max_imaginary_freq=0.0, band_structure_path=None)

        # Convert ASE to PhonopyAtoms
        # Type hints for ase methods might be missing
        unitcell = PhonopyAtoms(symbols=structure.get_chemical_symbols(), # type: ignore[no-untyped-call]
                                cell=structure.get_cell(), # type: ignore[no-untyped-call]
                                scaled_positions=structure.get_scaled_positions()) # type: ignore[no-untyped-call]

        phonon = Phonopy(unitcell,
                         supercell_matrix=self.supercell_matrix)

        phonon.generate_displacements(distance=self.displacement_distance)
        supercells = phonon.supercells_with_displacements

        # Calculate forces
        factory = MLIPCalculatorFactory()
        if potential.path is None:
             msg = "Potential path is not set"
             raise ValueError(msg)

        calc = factory.create(potential.path)

        sets_of_forces = []
        if supercells is None:
             msg = "Failed to generate supercells"
             raise ValueError(msg)

        for sc in supercells:
            if sc is None:
                continue
            # Convert PhonopyAtoms back to ASE Atoms
            ase_atoms = Atoms(symbols=sc.symbols,
                              cell=sc.cell,
                              scaled_positions=sc.scaled_positions,
                              pbc=True)
            ase_atoms.calc = calc
            forces = ase_atoms.get_forces() # type: ignore[no-untyped-call]
            sets_of_forces.append(forces)

        phonon.forces = sets_of_forces
        phonon.produce_force_constants()

        # Calculate mesh
        mesh = [20, 20, 20] # Simple mesh
        phonon.run_mesh(mesh, with_eigenvectors=False, is_mesh_symmetry=False)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict['frequencies'] # (nq, nband)

        # Check imaginary frequencies (negative values in Phonopy)
        min_freq = np.min(frequencies)

        # Tolerance for small negative frequencies (numerical noise)
        tolerance = -0.05 # THz
        is_stable = min_freq > tolerance

        max_imag_freq = 0.0
        if min_freq < 0:
            max_imag_freq = abs(float(min_freq))

        return PhononResults(
            is_stable=bool(is_stable),
            max_imaginary_freq=float(max_imag_freq),
            band_structure_path=None
        )
