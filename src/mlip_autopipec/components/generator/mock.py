import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from mlip_autopipec.components.generator.base import BaseGenerator
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    """
    Mock implementation of the Generator component.

    Generates random structures based on configured parameters (cell size, number of atoms, etc.).
    Useful for testing the pipeline flow without heavy computation.
    """

    def _extract_config(self, effective_config: dict[str, Any]) -> tuple[float, int, list[int]]:
        """Extracts configuration values, potentially raising conversion errors."""
        try:
            cell_size = float(effective_config["cell_size"])
            n_atoms = int(effective_config["n_atoms"])
            atomic_numbers = effective_config["atomic_numbers"]
        except (KeyError, ValueError, TypeError) as e:
            logger.exception("Configuration extraction failed")
            msg = f"Invalid generator configuration format: {e}"
            raise ValueError(msg) from e
        else:
            return cell_size, n_atoms, atomic_numbers

    def _validate_values(self, cell_size: float, n_atoms: int, atomic_numbers: list[int]) -> None:
        """Validates extracted values."""
        if cell_size <= 0:
            msg = f"Invalid cell_size: {cell_size}"
            raise ValueError(msg)

        if n_atoms <= 0:
            msg = f"Invalid n_atoms: {n_atoms}"
            raise ValueError(msg)

        if not atomic_numbers:
            msg = "atomic_numbers cannot be empty"
            raise ValueError(msg)

    def generate(
        self, n_structures: int, config: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        """
        Generate mock structures.

        Args:
            n_structures: The number of structures to generate.
            config: Optional runtime configuration override.

        Yields:
            Structure: Generated mock structure.

        Raises:
            ValueError: If configuration is invalid.
        """
        logger.info(f"Generating {n_structures} mock structures")

        if n_structures <= 0:
            logger.warning("n_structures must be positive")
            return

        # Merge configuration: method config overrides component config
        effective_config = self.config.model_dump()
        if config:
            effective_config.update(config)

        cell_size, n_atoms, atomic_numbers = self._extract_config(effective_config)
        self._validate_values(cell_size, n_atoms, atomic_numbers)

        # Ensure strict iterator behavior (no intermediate list)
        for _ in range(n_structures):
            pos = np.random.rand(n_atoms, 3) * cell_size
            numbers = np.array(atomic_numbers)
            cell = np.eye(3) * cell_size
            pbc = np.array([True, True, True])
            yield Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
