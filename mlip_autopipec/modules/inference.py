# ruff: noqa: D101, D102
"""Module for running inference simulations with LAMMPS."""

import logging
from typing import Generator, Union
from unittest.mock import MagicMock

from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config_schemas import SystemConfig

logger = logging.getLogger(__name__)


class LammpsRunner:
    """Manages and executes LAMMPS molecular dynamics simulations.

    This class is responsible for running MD simulations using a given MLIP.
    It is designed as a generator-based state machine that can be paused and
    resumed, yielding control back to the main application loop when it
    detects high model uncertainty.
    """

    def __init__(self, config: SystemConfig, potential_path: str) -> None:
        """Initialize the LammpsRunner.

        Args:
            config: The system configuration object.
            potential_path: The file path to the MLIP potential to be used.

        """
        if not config.inference:
            raise ValueError("Inference parameters must be defined in the config.")
        self.config = config
        self.potential_path = potential_path
        self._step = 0

        # This mock sequence simulates a run that is stable for a few steps,
        # then hits an uncertain structure, and would become stable again.
        # This allows testing the full loop logic.
        self._mock_uncertainty_counter = 0
        self._mock_uncertainty_sequence = [1.0, 1.5, 2.0, 4.5, 2.5]

    def _get_mock_extrapolation_grade(self, atoms: Atoms) -> float:
        """Mock the pacemaker uncertainty calculation for development.

        In a real implementation, this would involve a call to the `pacemaker`
        library to get the `extrapolation_grade`.

        Args:
            atoms: The current ASE Atoms object (unused in mock).

        Returns:
            A float representing the mock uncertainty score.

        """
        grade = self._mock_uncertainty_sequence[
            self._mock_uncertainty_counter % len(self._mock_uncertainty_sequence)
        ]
        self._mock_uncertainty_counter += 1
        logger.info(f"Mock uncertainty for step {self._step}: {grade}")
        return grade

    def run(self) -> Generator[Union[str, Atoms], None, None]:
        """Execute the LAMMPS simulation as a generator.

        This method runs the MD simulation step-by-step. After each step, it
        evaluates the model's uncertainty.

        Yields:
            - "stable": If the model uncertainty is below the configured threshold.
            - Atoms: An ASE Atoms object of the current structure if the uncertainty
                     exceeds the threshold, indicating that a new DFT calculation
                     is needed.

        """
        logger.info("Initializing LAMMPS simulation...")
        # The actual `lammps` library would be initialized here.
        # For now, we mock the LAMMPS instance to avoid the dependency.
        mock_lmp = MagicMock()

        # A dummy Atoms object is created to represent the simulation cell.
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        mock_lmp.get_natoms.return_value = len(atoms)

        assert self.config.inference is not None
        total_steps = self.config.inference.total_simulation_steps
        logger.info(f"Starting MD simulation for {total_steps} steps.")

        while self._step < total_steps:
            self._step += 1
            logger.debug(f"Running MD step {self._step}")

            # In a real scenario:
            # 1. `mock_lmp.command("run 1")` would execute a single MD step.
            # 2. The updated atomic positions would be extracted to the `atoms` object.

            extrapolation_grade = self._get_mock_extrapolation_grade(atoms)

            if extrapolation_grade >= self.config.inference.uncertainty_threshold:
                logger.warning(
                    f"Uncertainty threshold exceeded at step {self._step} "
                    f"(grade={extrapolation_grade:.2f}). Yielding structure."
                )
                yield atoms
                # The generator pauses here. When resumed, the loop continues.
                logger.info("Resuming simulation after DFT calculation.")
            else:
                yield "stable"

        logger.info("MD simulation completed successfully.")
