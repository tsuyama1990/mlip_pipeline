import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mlip_autopipec.components.trainer.base import BaseTrainer
from mlip_autopipec.domain_models.potential import Potential

if TYPE_CHECKING:
    from mlip_autopipec.core.dataset import Dataset

logger = logging.getLogger(__name__)


class MockTrainer(BaseTrainer):
    def train(
        self, dataset: "Dataset", workdir: Path, previous_potential: Potential | None = None
    ) -> Potential:
        logger.info(f"Training mock potential in {workdir}")
        output_path = workdir / "potential.yace"
        workdir.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return Potential(path=output_path, metrics={"rmse_energy": 0.01})

    def __repr__(self) -> str:
        return f"<MockTrainer(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"MockTrainer({self.name})"
