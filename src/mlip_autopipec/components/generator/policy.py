import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class GenerationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    builder_name: str
    n_structures: int = Field(gt=0)


class ExplorationPolicy:
    """Adaptive exploration policy for structure generation."""

    def decide_next_batch(
        self,
        current_cycle: int,
        current_metrics: dict[str, Any],
        n_total: int,
        ratios: dict[str, float] | None = None,
    ) -> list[GenerationTask]:
        """
        Decide the composition of the next batch of structures.

        Args:
            current_cycle: Current active learning cycle number.
            current_metrics: Metrics from previous cycles (e.g. validation errors).
            n_total: Total number of structures to generate.
            ratios: Optional dictionary of policy ratios (e.g. 'cycle0_bulk').

        Returns:
            List of GenerationTask objects specifying what to generate.
        """
        if n_total <= 0:
            return []

        if ratios is None:
            ratios = {}

        tasks: list[GenerationTask] = []

        # Cycle 0: Cold Start
        if current_cycle == 0:
            # Defaults: 60% Bulk, 40% Surface (from config or fallback)
            bulk_ratio = ratios.get("cycle0_bulk", 0.6)

            n_bulk = int(n_total * bulk_ratio)
            n_surface = n_total - n_bulk

            if n_bulk > 0:
                tasks.append(GenerationTask(builder_name="bulk", n_structures=n_bulk))
            if n_surface > 0:
                tasks.append(GenerationTask(builder_name="surface", n_structures=n_surface))

            logger.info(f"Cycle 0 Policy: {n_bulk} Bulk, {n_surface} Surface")
            return tasks

        # Cycle > 0: Adaptive
        # Check for specific weaknesses
        surface_error = 0.0
        val_metrics = current_metrics.get("validation_error", {})
        if isinstance(val_metrics, dict):
            surface_error = val_metrics.get("surface", 0.0)

        # Threshold for high error (e.g., 0.1 eV/A force RMSE or similar unitless metric)
        if surface_error > 0.1:
            logger.info(f"High surface error ({surface_error:.3f}) detected. Boosting surface sampling.")
            n_surface = int(n_total * 0.6)
            n_bulk = n_total - n_surface
        else:
            # Default balanced exploration
            n_bulk = int(n_total * 0.5)
            n_surface = n_total - n_bulk

        if n_bulk > 0:
            tasks.append(GenerationTask(builder_name="bulk", n_structures=n_bulk))
        if n_surface > 0:
            tasks.append(GenerationTask(builder_name="surface", n_structures=n_surface))

        logger.info(f"Adaptive Policy: {tasks}")
        return tasks
