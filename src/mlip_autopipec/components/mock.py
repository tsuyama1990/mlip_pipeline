import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms

from mlip_autopipec.components.base import BaseGenerator, BaseOracle, BaseTrainer
from mlip_autopipec.domain_models import Dataset, Potential, Structure

logger = logging.getLogger("mlip_autopipec")

class MockGenerator(BaseGenerator):
    """
    Mock implementation of a structure generator.
    """
    def generate(self, n_structures: int, cycle: int = 0, metrics: dict[str, Any] | None = None) -> Iterator[Structure]:
        logger.info(f"MockGenerator: Generating {n_structures} structures (cycle={cycle})")
        for i in range(n_structures):
            atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
            yield Structure(atoms=atoms, provenance="mock_generator", tags={"index": i, "cycle": cycle})

class MockOracle(BaseOracle):
    """
    Mock implementation of an oracle.
    """
    def compute(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        logger.info("MockOracle: Computing structures...")
        count = 0
        for structure in structures:
            # Add dummy results
            structure.atoms.info["energy"] = -1.0
            structure.atoms.arrays["forces"] = np.array([[0.0, 0.0, 0.0]] * len(structure.atoms))
            structure.provenance = "mock_oracle"
            count += 1
            yield structure
        logger.info(f"MockOracle: Computed {count} structures")

class MockTrainer(BaseTrainer):
    """
    Mock implementation of a trainer.
    """
    def train(self, dataset: Dataset) -> Potential:
        logger.info(f"MockTrainer: Training on dataset {dataset.path}")
        # Return a dummy potential path
        return Potential(path=dataset.path.with_suffix(".yace"), type="mock")
