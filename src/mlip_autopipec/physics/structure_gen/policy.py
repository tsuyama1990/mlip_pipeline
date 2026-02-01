from typing import Protocol

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.exploration import ExplorationTask


class PolicyStrategy(Protocol):
    """Protocol for exploration policy strategies."""

    def decide(self, config: Config) -> ExplorationTask:
        """Decide the exploration task based on configuration."""
        ...


class MetalPolicy:
    """Policy for metallic systems (e.g. alloys)."""

    def decide(self, config: Config) -> ExplorationTask:
        return ExplorationTask(
            method="MD",
            modifiers=["swap"],
            temperature=config.md.temperature,
            steps=config.md.n_steps,
        )


class InsulatorPolicy:
    """Policy for insulating/ceramic systems."""

    def decide(self, config: Config) -> ExplorationTask:
        return ExplorationTask(
            method="Static",
            modifiers=["defect"]
        )


class AdaptivePolicy:
    """
    The Decision Engine for the Exploration Phase.
    Decides the strategy based on cycle count and material configuration.
    """

    def decide(self, cycle: int, config: Config) -> ExplorationTask:
        """
        Decide the exploration task.

        Args:
            cycle: Current generation number.
            config: Global configuration.

        Returns:
            ExplorationTask describing what to do.
        """
        # Cycle 0: Cold Start
        if cycle == 0:
            return ExplorationTask(
                method="Static",
                modifiers=["strain", "rattle"]
            )

        # Dispatch to specific policy
        strategy: PolicyStrategy
        if config.policy.is_metal:
            strategy = MetalPolicy()
        else:
            strategy = InsulatorPolicy()

        return strategy.decide(config)
