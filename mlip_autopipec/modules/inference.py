"""Module for running inference simulations with LAMMPS."""

import logging
from collections.abc import Generator
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
        self._mock_counter = 0
        self._mock_sequence = [1.0, 1.5, 2.0, 4.5, 2.5]

    def get_extrapolation_grade(self, atoms: Atoms) -> float:
        """Mock the pacemaker uncertainty calculation for development."""
        grade = self._mock_sequence[self._mock_counter % len(self._mock_sequence)]
        self._mock_counter += 1
        logger.info("Mock uncertainty grade: %s", grade)
        return grade


class LammpsRunner:
    """Manages and executes LAMMPS molecular dynamics simulations."""

    def __init__(
        self,
        config: SystemConfig,
        potential_path: str,
        quantifier: UncertaintyQuantifier,
    ) -> None:
        """Initialize the LammpsRunner."""
        if not config.inference:
            msg = "Inference parameters must be defined in the config."
            raise ValueError(msg)
        self.config = config
        self.potential_path = potential_path
        self.quantifier = quantifier
        self._step = 0

    def _extract_periodic_subcell(
        self, atoms: Atoms, uncertain_atom_index: int, rcut: float, delta_buffer: float
    ) -> Atoms:
        """Extract a periodic sub-cell centered on the uncertain atom."""
        box_size = 2 * (rcut + delta_buffer)
        displacements = atoms.get_distances(
            uncertain_atom_index, np.arange(len(atoms)), mic=True, vector=True
        )
        in_box_mask = np.all(np.abs(displacements) < box_size / 2.0, axis=1)
        indices = np.where(in_box_mask)[0]
        new_cell = [box_size, box_size, box_size]
        new_positions = displacements[indices] + box_size / 2.0
        symbols = np.array(atoms.get_chemical_symbols())
        subcell_atoms = Atoms(
            symbols=symbols[indices],
            positions=new_positions,
            cell=new_cell,
            pbc=True,
        )
        return subcell_atoms

    def _generate_force_mask(self, subcell_atoms: Atoms, rcut: float) -> np.ndarray:
        """Generate a force mask for the sub-cell."""
        center = np.diag(subcell_atoms.get_cell()) / 2.0
        distances = np.linalg.norm(subcell_atoms.positions - center, axis=1)
        mask = np.where(distances < rcut, 1.0, 0.0)
        return np.repeat(mask.reshape(-1, 1), 3, axis=1)

    def run(self) -> Generator[tuple[Atoms, np.ndarray], None, None]:
        """Execute the LAMMPS simulation, yielding uncertain structures."""
        logger.info("Initializing LAMMPS simulation...")
        mock_lmp = MagicMock()
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        mock_lmp.get_natoms.return_value = len(atoms)

        assert self.config.inference is not None
        total_steps = self.config.inference.total_simulation_steps
        logger.info("Starting MD simulation for %s steps.", total_steps)

        while self._step < total_steps:
            self._step += 1
            logger.debug("Running MD step %s", self._step)

            try:
                extrapolation_grade = self.quantifier.get_extrapolation_grade(atoms)
                if extrapolation_grade >= self.config.inference.uncertainty_threshold:
                    logger.warning(
                        "High uncertainty detected (grade=%.2f) at step %d!",
                        extrapolation_grade,
                        self._step,
                    )
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
                    return

            except Exception as e:
                logger.exception("An error occurred during MD step %s", self._step)
                msg = f"LAMMPS simulation failed at step {self._step}"
                raise RuntimeError(msg) from e

        logger.info("MD simulation finished without exceeding uncertainty threshold.")
