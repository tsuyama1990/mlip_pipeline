import logging
from collections.abc import Iterable
from pathlib import Path

from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.interfaces import BaseTrainer

logger = logging.getLogger(__name__)


class MockTrainer(BaseTrainer):
    """
    Mock implementation of a potential trainer.
    Consumes the dataset and returns a dummy Potential.
    """

    def train(
        self,
        dataset: Iterable[Structure],
        initial_potential: Potential | None = None,
        workdir: Path | None = None,
    ) -> Potential:
        """
        Simulates training by iterating over the dataset and creating a dummy potential file.
        """
        logger.info("MockTrainer: Training potential...")

        # Consume the dataset to simulate work and verify iteration
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
