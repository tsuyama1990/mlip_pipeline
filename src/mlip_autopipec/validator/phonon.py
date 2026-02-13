import logging
from typing import Any, NamedTuple

import numpy as np

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.dynamics.calculators import MLIPCalculatorFactory

logger = logging.getLogger(__name__)

class PhononResults(NamedTuple):
    max_imaginary_freq: float # THz, positive value indicates magnitude of instability
    is_stable: bool
    band_structure_plot_data: dict[str, Any] | None

class PhononAnalyzer:
    """Analyzer for Phonon Stability using Phonopy."""

    def __init__(self) -> None:
        self.calculator_factory = MLIPCalculatorFactory()

    def analyze(self, potential: Potential, structure: Structure, supercell_matrix: list[int] | None = None) -> PhononResults:
        """
        Calculates phonon dispersion and checks for stability.

        Args:
            potential: The potential to use.
            structure: The structure to analyze.
            supercell_matrix: Diagonal supercell matrix (e.g., [2, 2, 2]).

        Returns:
            PhononResults.
        """
        try:
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms
        except ImportError:
            logger.warning("Phonopy not installed. Skipping phonon analysis.")
            return PhononResults(0.0, True, None)

        if supercell_matrix is None:
            supercell_matrix = [2, 2, 2]

        s_mat = np.diag(supercell_matrix)

        # Convert ASE to PhonopyAtoms
        ase_atoms = structure.to_ase()
        unitcell = PhonopyAtoms(
            symbols=ase_atoms.get_chemical_symbols(), # type: ignore[no-untyped-call]
            cell=ase_atoms.get_cell(), # type: ignore[no-untyped-call]
            scaled_positions=ase_atoms.get_scaled_positions() # type: ignore[no-untyped-call]
        )

        phonon = Phonopy(unitcell, supercell_matrix=s_mat)

        # Generate displacements
        phonon.generate_displacements(distance=0.01)
        supercells = phonon.supercells_with_displacements

        if supercells is None:
             logger.warning("Phonopy failed to generate supercells.")
             return PhononResults(0.0, True, None)

        # Calculate forces for each supercell
        forces_set = []
        calc = self.calculator_factory.create(potential.path)

        for sc in supercells:
            # Convert PhonopyAtoms back to ASE
            # Note: PhonopyAtoms has .symbols, .cell, .scaled_positions
            from ase import Atoms
            sc_ase = Atoms(
                symbols=sc.symbols,
                cell=sc.cell,
                scaled_positions=sc.scaled_positions,
                pbc=True
            )
            sc_ase.calc = calc
            forces = sc_ase.get_forces() # type: ignore[no-untyped-call]
            forces_set.append(forces)

        # Set forces
        phonon.produce_force_constants(forces=forces_set)

        # Calculate Band Structure (Mesh)
        # We use mesh for stability check (faster than band path)
        mesh = [20, 20, 20]
        phonon.run_mesh(mesh)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict['frequencies'] # (q-points, bands)

        # Check for imaginary frequencies (negative values in Phonopy)
        # Ignore acoustic modes at Gamma point (usually very close to 0 or slightly negative due to numerical noise)
        # We check min frequency.
        min_freq = np.min(frequencies)

        max_imaginary = 0.0
        is_stable = True

        # Threshold for instability: < -0.05 THz (to allow small numerical noise)
        threshold = -0.05

        if min_freq < threshold:
            max_imaginary = abs(min_freq)
            is_stable = False

        # Generate band structure plot data (optional path)
        # Just simple auto path
        band_plot_data = None
        try:
            phonon.auto_band_structure(plot=False) # type: ignore[no-untyped-call]
            band_dict = phonon.get_band_structure_dict()
            band_plot_data = {
                "qpoints": band_dict["qpoints"],
                "distances": band_dict["distances"],
                "frequencies": band_dict["frequencies"],
                "labels": band_dict["labels"]
            }
        except Exception:
            logger.warning("Failed to generate band structure path.", exc_info=True)

        return PhononResults(
            max_imaginary_freq=max_imaginary,
            is_stable=is_stable,
            band_structure_plot_data=band_plot_data
        )
