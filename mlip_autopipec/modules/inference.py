# ruff: noqa: D101, D102
"""Module for running inference simulations with LAMMPS."""

import logging
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config_schemas import SystemConfig

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """Handles the uncertainty quantification for a given structure."""

    def __init__(self) -> None:
        """Initialize the UncertaintyQuantifier."""
        # This mock sequence simulates a run that is stable for a few steps,
        # then hits an uncertain structure, and would become stable again.
        self._mock_counter = 0
        self._mock_sequence = [1.0, 1.5, 2.0, 4.5, 2.5]

    def get_extrapolation_grade(self, atoms: Atoms) -> float:
        """Mock the pacemaker uncertainty calculation for development.

        In a real implementation, this would involve a call to the `pacemaker`
        library.

        Args:
            atoms: The current ASE Atoms object (unused in mock).

        Returns:
            A float representing the mock uncertainty score.

        """
        grade = self._mock_sequence[self._mock_counter % len(self._mock_sequence)]
        self._mock_counter += 1
        logger.info(f"Mock uncertainty grade: {grade}")
        return grade


class LammpsRunner:
    """Manages and executes LAMMPS molecular dynamics simulations.

    This class is the core of the active learning inference engine. It runs an
    MD simulation using a given machine-learned potential and uses an
    `UncertaintyQuantifier` to check the model's extrapolation grade at each
    step.

    When a structure with high uncertainty is detected, this class is responsible
    for executing the "periodic embedding" algorithm. Instead of passing the
    entire large simulation cell to the expensive DFT calculation, it extracts a
    small, fully periodic sub-cell centered on the atom causing the uncertainty.

    To prevent unphysical forces at the boundaries of this small cell from
    contaminating the training data, it also generates a "force mask," a
    per-atom weight vector that tells the training engine to ignore forces on
    the atoms in the buffer region of the sub-cell.
    """

    def __init__(
        self,
        config: SystemConfig,
        potential_path: str,
        quantifier: UncertaintyQuantifier,
    ) -> None:
        """Initialize the LammpsRunner.

        Args:
            config: The system configuration object.
            potential_path: The file path to the MLIP potential to be used.
            quantifier: An object to calculate model uncertainty.

        """
        if not config.inference:
            raise ValueError("Inference parameters must be defined in the config.")
        self.config = config
        self.potential_path = potential_path
        self.quantifier = quantifier
        self._step = 0

    def _extract_periodic_subcell(
        self, atoms: Atoms, uncertain_atom_index: int, rcut: float, delta_buffer: float
    ) -> Atoms:
        """Extract a periodic sub-cell centered on the uncertain atom.

        This function implements the periodic embedding algorithm. It carves out
        a cubic, periodic box of atoms centered around a specific atom of
        interest from a larger simulation cell. It correctly handles periodic
        boundary conditions, wrapping atoms from the other side of the original
        cell into the new one.

        Args:
            atoms: The original, large ASE Atoms object.
            uncertain_atom_index: The index of the atom to center the box on.
            rcut: The cutoff radius defining the "core" region of the sub-cell.
            delta_buffer: An additional buffer size to add to the cutoff,
                          defining the total size of the sub-cell.

        Returns:
            A new, smaller, fully periodic ASE Atoms object.

        """
        box_size = 2 * (rcut + delta_buffer)

        # Get displacements from the uncertain atom using minimum image convention
        displacements = atoms.get_distances(  # type: ignore[no-untyped-call]
            uncertain_atom_index, np.arange(len(atoms)), mic=True, vector=True
        )

        # Find atoms within the cubic box
        in_box_mask = np.all(np.abs(displacements) < box_size / 2.0, axis=1)
        indices = np.where(in_box_mask)[0]

        # Create the new Atoms object
        new_cell = [box_size, box_size, box_size]
        new_positions = displacements[indices] + box_size / 2.0

        subcell_atoms = Atoms(
            symbols=np.array(atoms.get_chemical_symbols())[indices],  # type: ignore[no-untyped-call]
            positions=new_positions,
            cell=new_cell,
            pbc=True,
        )

        return subcell_atoms

    def _generate_force_mask(self, subcell_atoms: Atoms, rcut: float) -> np.ndarray:
        """Generate a force mask for the sub-cell.

        This function creates a per-atom mask that is 1.0 for atoms inside the
        "core" region (defined by `rcut`) and 0.0 for atoms in the "buffer"
        region. The output is an (N, 3) array, suitable for direct multiplication
        with a forces array.

        Args:
            subcell_atoms: The sub-cell ASE Atoms object.
            rcut: The cutoff radius defining the core region.

        Returns:
            An (N, 3) NumPy array where N is the number of atoms.

        """
        center = np.diag(subcell_atoms.get_cell()) / 2.0  # type: ignore[no-untyped-call]
        distances = np.linalg.norm(subcell_atoms.positions - center, axis=1)
        mask = np.where(distances < rcut, 1.0, 0.0)
        # Repeat the mask for x, y, z components of the force
        return np.repeat(mask.reshape(-1, 1), 3, axis=1)

    def run(self) -> Generator[tuple[Atoms, np.ndarray], None, None]:
        """Execute the LAMMPS simulation, yielding uncertain structures.

        This method runs the MD simulation step-by-step. If a structure with
        high uncertainty is found, it performs periodic embedding and yields the
        sub-cell and its corresponding force mask. The generator then terminates.

        Yields:
            A tuple containing:
            - embedded_atoms (ase.Atoms): The smaller, periodic sub-cell.
            - force_mask (np.ndarray): An (N, 3) array of weights (0.0 or 1.0).

        """
        logger.info("Initializing LAMMPS simulation...")
        mock_lmp = MagicMock()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        mock_lmp.get_natoms.return_value = len(atoms)

        assert self.config.inference is not None
        total_steps = self.config.inference.total_simulation_steps
        logger.info(f"Starting MD simulation for {total_steps} steps.")

        while self._step < total_steps:
            self._step += 1
            logger.debug(f"Running MD step {self._step}")

            try:
                # In a real scenario, this would run a single MD step.
                # mock_lmp.command("run 1")

                extrapolation_grade = self.quantifier.get_extrapolation_grade(atoms)
                if extrapolation_grade >= self.config.inference.uncertainty_threshold:
                    logger.warning(
                        "High uncertainty detected (grade=%.2f) at step %d!",
                        extrapolation_grade,
                        self._step,
                    )
                    # For this mock, we'll assume the first atom is the cause.
                    uncertain_atom_index = 0
                    embedded_atoms = self._extract_periodic_subcell(
                        atoms=atoms,
                        uncertain_atom_index=uncertain_atom_index,
                        rcut=self.config.inference.embedding_rcut,
                        delta_buffer=self.config.inference.embedding_delta_buffer,
                    )
                    force_mask = self._generate_force_mask(
                        subcell_atoms=embedded_atoms,
                        rcut=self.config.inference.embedding_rcut,
                    )
                    yield embedded_atoms, force_mask
                    return  # Stop the generator after finding an uncertain structure

            except Exception as e:
                logger.error(f"An error occurred during MD step {self._step}: {e}")
                raise RuntimeError(
                    f"LAMMPS simulation failed at step {self._step}"
                ) from e

        logger.info("MD simulation finished without exceeding uncertainty threshold.")
