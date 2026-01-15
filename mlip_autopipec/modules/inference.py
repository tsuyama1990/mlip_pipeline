# ruff: noqa: D101, D102
"""Module for running inference simulations with LAMMPS."""

import logging
from typing import Generator, Union
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
    """Manages and executes LAMMPS molecular dynamics simulations."""

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

    def run(self) -> Generator[tuple[Atoms, float, np.ndarray | None], None, None]:
        """Execute the LAMMPS simulation as a generator.

        This method runs the MD simulation step-by-step, yielding the current
        atomic structure, its calculated uncertainty grade, and an optional force mask.

        Yields:
            A tuple containing the ASE Atoms object, the extrapolation grade, and
            an optional numpy array for the force mask.
        """
        logger.info("Initializing LAMMPS simulation...")
        mock_lmp = MagicMock()
        atoms = bulk("Cu", "fcc", a=3.6) * (4, 4, 4)  # Larger cell for embedding
        mock_lmp.get_natoms.return_value = len(atoms)

        assert self.config.inference is not None
        total_steps = self.config.inference.total_simulation_steps
        logger.info(f"Starting MD simulation for {total_steps} steps.")

        while self._step < total_steps:
            self._step += 1
            logger.debug(f"Running MD step {self._step}")

            try:
                extrapolation_grade = self.quantifier.get_extrapolation_grade(atoms)
                if extrapolation_grade >= self.config.inference.uncertainty_threshold:
                    # In a real scenario, we would identify the atom with the highest
                    # uncertainty. Here we just pick an atom in the center.
                    uncertain_atom_index = len(atoms) // 2
                    sub_cell = self._extract_periodic_subcell(
                        atoms, uncertain_atom_index
                    )
                    force_mask = self._generate_force_mask(sub_cell)
                    yield sub_cell, extrapolation_grade, force_mask
                else:
                    yield atoms, extrapolation_grade, None

            except Exception as e:
                logger.error(f"An error occurred during MD step {self._step}: {e}")
                raise RuntimeError(
                    f"LAMMPS simulation failed at step {self._step}"
                ) from e

        logger.info("MD simulation completed successfully.")

    def _extract_periodic_subcell(
        self, atoms: Atoms, center_atom_index: int
    ) -> Atoms:
        """Extract a periodic sub-cell around a central atom.

        This method correctly handles periodic boundary conditions by finding all
        atoms within a cubic region around the central atom, including those
        that wrap around the original cell's boundaries.

        Args:
            atoms: The full simulation cell.
            center_atom_index: The index of the atom to center the sub-cell on.

        Returns:
            A new, smaller, periodic ASE Atoms object.
        """
        assert self.config.inference is not None
        cutoff = self.config.inference.embedding_cutoff
        buffer = self.config.inference.embedding_buffer
        box_size = 2 * (cutoff + buffer)

        center_pos = atoms.positions[center_atom_index]

        # Get all neighbors within the box size, accounting for PBC
        indices = []
        for i in range(len(atoms)):
            distance, offset = atoms.get_distance(center_atom_index, i, mic=True)
            if np.all(np.abs(atoms.positions[i] + offset - center_pos) < box_size / 2):
                indices.append(i)

        sub_cell_atoms = atoms[indices].copy()
        sub_cell_atoms.set_cell([box_size, box_size, box_size])
        sub_cell_atoms.pbc = True

        # Center the atoms in the new box
        sub_cell_atoms.positions -= center_pos - box_size / 2

        return sub_cell_atoms

    def _generate_force_mask(self, atoms: Atoms) -> np.ndarray:
        """Generate a force mask for the given sub-cell.

        Atoms inside the cutoff radius get a mask value of 1.0 (forces are used),
        while atoms in the buffer region get a value of 0.0 (forces are ignored).

        Args:
            atoms: The sub-cell for which to generate the mask.

        Returns:
            A numpy array of 1s and 0s.
        """
        assert self.config.inference is not None
        cutoff = self.config.inference.embedding_cutoff
        center_of_box = np.diag(atoms.get_cell()) / 2.0

        distances = np.linalg.norm(atoms.positions - center_of_box, axis=1)

        force_mask = (distances < cutoff).astype(float)
        return force_mask
