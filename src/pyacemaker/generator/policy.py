"""Adaptive policy for selecting exploration strategies."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.generator.strategies import (
    ExplorationStrategy,
    M3GNetStrategy,
    RandomStrategy,
)


class ExplorationContext(BaseModel):
    """Context for making exploration decisions."""

    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(..., description="Current active learning cycle index")
    seed_structure: StructureMetadata | None = Field(
        default=None, description="Current seed structure metadata"
    )
    uncertainty_state: Any = Field(
        default=None, description="Uncertainty state of the model (optional)"
    )


class AdaptivePolicy:
    """Decides which exploration strategy to use based on context."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize with generator configuration."""
        self.config = config.structure_generator

    def decide_strategy(self, context: ExplorationContext) -> ExplorationStrategy:
        """Select the optimal exploration strategy based on context."""

        # Cold Start Logic
        if context.cycle == 0:
            if self.config.initial_exploration == "m3gnet":
                return M3GNetStrategy()
            # If random, or fallback
            return RandomStrategy(
                strain_range=self.config.strain_range, rattle_amplitude=self.config.rattle_amplitude
            )

        # High Uncertainty Logic (Example)
        if context.uncertainty_state:
            # Check for high uncertainty
            # Assuming uncertainty_state object has gamma_max attribute
            gamma_max = getattr(context.uncertainty_state, "gamma_max", None)
            if gamma_max is not None and gamma_max > 5.0:  # Threshold could be in config
                # Use cautious strategy (Random with smaller perturbation)
                return RandomStrategy(
                    strain_range=self.config.strain_range * 0.5,
                    rattle_amplitude=self.config.rattle_amplitude * 0.5,
                )

        # Default Strategy
        return RandomStrategy(
            strain_range=self.config.strain_range, rattle_amplitude=self.config.rattle_amplitude
        )
