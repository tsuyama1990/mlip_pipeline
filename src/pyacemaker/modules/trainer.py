"""Trainer (Pacemaker) module implementation."""

import tempfile
from pathlib import Path

from pyacemaker.core.base import ModuleResult
from pyacemaker.core.interfaces import Trainer
from pyacemaker.domain_models.models import (
    ActiveSet,
    Potential,
    PotentialType,
    StructureMetadata,
)


class PacemakerTrainer(Trainer):
    """Pacemaker trainer implementation."""

    def run(self) -> ModuleResult:
        """Run the trainer."""
        self.logger.info("Running PacemakerTrainer")
        return ModuleResult(status="success")

    def train(
        self, dataset: list[StructureMetadata], initial_potential: Potential | None = None
    ) -> Potential:
        """Train a potential."""
        self.logger.info(f"Training on dataset of size {len(dataset)} (mock)")

        # Create a dummy potential file
        # Use NamedTemporaryFile to avoid hardcoded paths
        with tempfile.NamedTemporaryFile(suffix=".yace", delete=False) as f:
            pot_path = Path(f.name)
            f.write(b"mock_potential_data")

        return Potential(
            path=pot_path,
            type=PotentialType.PACE,
            version="1.0",
            metrics={"rmse_energy": 0.005},
        )

    def select_active_set(self, candidates: list[StructureMetadata], n_select: int) -> ActiveSet:
        """Select active set."""
        self.logger.info(f"Selecting {n_select} from {len(candidates)} candidates (mock)")

        # Select first n_select as a dummy strategy
        selected = candidates[:n_select]

        return ActiveSet(
            structure_ids=[s.id for s in selected],
            selection_criteria="mock_selection",
        )
