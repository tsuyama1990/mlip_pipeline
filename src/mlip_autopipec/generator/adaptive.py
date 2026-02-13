import logging
from collections.abc import Iterator
from typing import Any

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.generator.interface import BaseGenerator
from mlip_autopipec.generator.m3gnet_gen import M3GNetGenerator
from mlip_autopipec.generator.random_gen import RandomGenerator

logger = logging.getLogger(__name__)


class AdaptiveGenerator(BaseGenerator):
    """Adaptive Generator that selects strategy based on cycle and policy."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.policy = self.config.policy

        # Initialize RandomGenerator if seed is available, to generate mock MD structures
        self.random_gen: RandomGenerator | None = None
        if self.config.seed_structure_path:
            try:
                self.random_gen = RandomGenerator(config)
            except Exception as e:
                logger.warning(f"Failed to initialize internal RandomGenerator: {e}")

        # Fallback generator
        self.m3gnet_gen = M3GNetGenerator(config)

    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        cycle = context.get("cycle", 0)

        # Temperature Schedule
        schedule = self.policy.temperature_schedule
        if not schedule:
            schedule = [300.0]

        # Pick temperature based on cycle (cycle 0 -> index 0, etc.)
        idx = min(cycle, len(schedule) - 1)
        temperature = schedule[idx]

        logger.info(f"AdaptiveGenerator: Cycle {cycle}, using Temperature={temperature}K")

        count = context.get("count", self.config.mock_count)

        # For Cycle 02, we mock the MD execution by using RandomGenerator.
        # We accept the RandomGenerator output and re-label it.

        if self.random_gen:
            for s in self.random_gen.explore({"count": count}):
                s.provenance = f"md_{temperature}K"
                yield s
        else:
            logger.info(
                "No seed structure provided. Falling back to M3GNetGenerator for initial structures."
            )
            yield from self.m3gnet_gen.explore(context)
