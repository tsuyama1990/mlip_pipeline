import logging
import secrets
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.interfaces import BaseTrainer

logger = logging.getLogger(__name__)


class MockTrainer(BaseTrainer):
    """
    Mock implementation of a potential trainer.
    Consumes the dataset and returns a dummy Potential.
    """

    def __init__(self, fail_rate: float = 0.0, **kwargs: Any) -> None:
        """
        Args:
            fail_rate: Probability of failure during training (0.0 to 1.0).
            **kwargs: Ignored extra arguments.
        """
        self.fail_rate = fail_rate
        self.rng = np.random.default_rng(secrets.randbits(128))
        if kwargs:
            logger.debug(f"MockTrainer received extra args: {kwargs}")

    def train(
        self,
        dataset: Iterable[Structure],
        initial_potential: Potential | None = None,
        workdir: Path | None = None,
    ) -> Potential:
        """
        Simulates training by iterating over the dataset and creating a dummy potential file.
        """
        if self.rng.random() < self.fail_rate:
            msg = "MockTrainer: Simulated failure during training."
            raise RuntimeError(msg)

        logger.info("MockTrainer: Training potential...")

        # Consume the dataset to simulate work and verify iteration
        # Streaming count to avoid OOM
        count = 0
        for _ in dataset:
            count += 1

        logger.info(f"MockTrainer: Processed {count} structures during training.")

        # Determine potential path
        if workdir:
            potential_path = workdir / "mock_potential.yace"
            # Ensure workdir exists (though orchestrator should handle this)
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            potential_path = Path("mock_potential.yace")

        # Create dummy file
        potential_path.touch()

        return Potential(
            path=potential_path,
            version="mock-v1",
            metrics={"train_mae_e": 0.001, "train_mae_f": 0.01},
        )
