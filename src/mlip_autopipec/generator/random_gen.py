import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.generator.interface import BaseGenerator

logger = logging.getLogger(__name__)


class RandomGenerator(BaseGenerator):
    """Generates random structures by rattling a seed structure."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        if not self.config.seed_structure_path:
            msg = "RandomGenerator requires a seed structure path in config."
            raise ValueError(msg)

        self.seed_path = self.config.seed_structure_path

    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        count = context.get("count", self.config.mock_count)

        if not self.seed_path.exists():
            msg = f"Seed structure file not found: {self.seed_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not self.seed_path.is_file():
            msg = f"Seed structure path is not a file: {self.seed_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"RandomGenerator: Generating {count} structures from {self.seed_path}")

        try:
            # Using iread to avoid loading everything if it's a large file.
            iterator = iread(str(self.seed_path))
            try:
                # Get the first structure
                seed_atoms = next(iterator)
            except StopIteration as e:
                msg = f"Seed file is empty: {self.seed_path}"
                raise ValueError(msg) from e
            except Exception as e:
                # iread might raise UnknownFileTypeError or others directly if file is invalid/empty
                # We raise ValueError for consistency
                msg = f"Failed to read seed structure from {self.seed_path}: {e}"
                raise ValueError(msg) from e

        except Exception as e:
            # If it's already ValueError, let it bubble, otherwise log and re-raise
            if not isinstance(e, ValueError):
                logger.exception("Failed to load seed structure")
            raise

        # Ensure it is Atoms
        if not isinstance(seed_atoms, Atoms):
            msg = f"Seed structure is not an ASE Atoms object: {type(seed_atoms)}"
            raise TypeError(msg)

        strain_range = self.config.policy.strain_range

        for _ in range(count):
            atoms = seed_atoms.copy()  # type: ignore[no-untyped-call]

            # Apply random strain
            if strain_range > 0:
                strain = np.random.uniform(-strain_range, strain_range, (3, 3))
                deformation = np.eye(3) + strain
                # atoms.cell is a Cell object, can be treated as 3x3 array for multiplication
                new_cell = atoms.cell @ deformation
                atoms.set_cell(new_cell, scale_atoms=True)  # type: ignore[no-untyped-call]

            # Rattle positions (displace atoms)
            atoms.rattle(stdev=0.1)

            yield Structure(atoms=atoms, provenance="random", label_status="unlabeled")
