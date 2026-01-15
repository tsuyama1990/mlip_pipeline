# ruff: noqa: D101, D102
"""Module for running inference simulations with LAMMPS."""

import logging
from typing import Generator, Union
from unittest.mock import MagicMock

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

    def run(self) -> Generator[tuple[Atoms, float], None, None]:
        """Execute the LAMMPS simulation as a generator.

        This method runs the MD simulation step-by-step, yielding the current
        atomic structure and its calculated uncertainty grade at each step.

        Yields:
            A tuple containing the ASE Atoms object and the extrapolation grade.
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
                yield atoms, extrapolation_grade

            except Exception as e:
                logger.error(f"An error occurred during MD step {self._step}: {e}")
                raise RuntimeError(
                    f"LAMMPS simulation failed at step {self._step}"
                ) from e

        logger.info("MD simulation completed successfully.")
