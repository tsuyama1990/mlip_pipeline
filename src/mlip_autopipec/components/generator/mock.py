import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from mlip_autopipec.components.generator.base import BaseGenerator
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class MockGenerator(BaseGenerator):
    def _validate_config(self, effective_config: dict[str, Any]) -> tuple[float, int, list[int]]:
        try:
            cell_size = float(effective_config["cell_size"])
            if cell_size <= 0:
                msg = f"Invalid cell_size: {cell_size}"
                raise ValueError(msg)

            n_atoms = int(effective_config["n_atoms"])
            if n_atoms <= 0:
                msg = f"Invalid n_atoms: {n_atoms}"
                raise ValueError(msg)

            atomic_numbers = effective_config["atomic_numbers"]
            if not atomic_numbers:
                msg = "atomic_numbers cannot be empty"
                raise ValueError(msg)

            return cell_size, n_atoms, atomic_numbers

        except (KeyError, ValueError, TypeError) as e:
            logger.exception("Configuration validation failed")
            msg = f"Invalid generator configuration: {e}"
            raise ValueError(msg) from e

    def generate(
        self, n_structures: int, config: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        logger.info(f"Generating {n_structures} mock structures")

        if n_structures <= 0:
            logger.warning("n_structures must be positive")
            return

        # Merge configuration: method config overrides component config
        effective_config = self.config.model_dump()
        if config:
            effective_config.update(config)

        cell_size, n_atoms, atomic_numbers = self._validate_config(effective_config)

        # Ensure strict iterator behavior (no intermediate list)
        for _ in range(n_structures):
            pos = np.random.rand(n_atoms, 3) * cell_size
            numbers = np.array(atomic_numbers)
            cell = np.eye(3) * cell_size
            pbc = np.array([True, True, True])
            yield Structure(positions=pos, atomic_numbers=numbers, cell=cell, pbc=pbc)
