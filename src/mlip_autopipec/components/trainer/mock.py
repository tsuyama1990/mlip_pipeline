import logging
import secrets
from pathlib import Path
from typing import Iterable, Optional, Any

import numpy as np

from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.interfaces import BaseTrainer

logger = logging.getLogger(__name__)


class MockTrainer(BaseTrainer):
    """
    Mock implementation of a potential trainer.
    Consumes the dataset and returns a dummy Potential.
    """

    def __init__(self, fail_rate: float = 0.0, output_name: str = "mock_potential", extension: str = ".yace", **kwargs: Any) -> None:
        """
        Args:
            fail_rate: Probability of failure during training (0.0 to 1.0).
            output_name: Base name for the output potential file.
            extension: File extension for the output potential file.
            **kwargs: Ignored extra arguments.
        """
        self.fail_rate = fail_rate
        self.output_name = output_name
        self.extension = extension
        self.rng = np.random.default_rng(secrets.randbits(128))
        if kwargs:
            logger.debug(f"MockTrainer received extra args: {kwargs}")

    def train(
        self,
        dataset: Iterable[Structure],
        initial_potential: Optional[Potential] = None,
        workdir: Optional[Path] = None,
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
        # Hard safeguard against infinite loops if an infinite iterator is passed
        max_structures = 1_000_000
        for _ in dataset:
            count += 1
            if count >= max_structures:
                logger.warning(f"MockTrainer: Reached safety limit of {max_structures} structures. Stopping consumption.")
                break
            # Simulate "work"
            if count % 1000 == 0:
                pass

        logger.info(f"MockTrainer: Processed {count} structures during training.")

        # Determine potential path
        filename = f"{self.output_name}{self.extension}"
        if workdir:
            potential_path = workdir / filename
            # Ensure workdir exists (though orchestrator should handle this)
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            potential_path = Path(filename)

        # Create dummy file
        potential_path.touch()

        return Potential(
            path=potential_path,
            version="mock-v1",
            metrics={"train_mae_e": 0.001, "train_mae_f": 0.01},
        )
