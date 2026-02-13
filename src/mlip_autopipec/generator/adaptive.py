import contextlib
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.datastructures import Structure
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

    @contextlib.contextmanager
    def _lammps_input_context(self, temperature: float, steps: int) -> Iterator[Path]:
        """
        Context manager that generates a LAMMPS input script and ensures cleanup.

        Yields:
            Path to the generated input file.
        """
        template = self.policy.lammps_template
        content = template.format(temperature=temperature, steps=steps)

        # Create a named temporary file
        # usage of delete=False is necessary to close the file and let other processes (LAMMPS) read it by path.
        # We ensure cleanup in finally block.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as tmp:
            logger.debug(f"Writing LAMMPS input content:\n{content}")
            tmp.write(content)
            path = Path(tmp.name)

        # File is closed here, but persists on disk due to delete=False
        try:
            logger.debug(f"Generated LAMMPS input file at {path}")
            yield path
        finally:
            # Ensure cleanup
            if path.exists():
                path.unlink()
            logger.debug(f"Cleaned up LAMMPS input file at {path}")

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

        # Use context manager for LAMMPS script lifecycle
        with self._lammps_input_context(temperature, self.policy.md_steps) as lammps_script_path:
            # In a real scenario, we would pass this path to the Dynamics engine.
            # For now, we just log it.
            logger.debug(f"Using LAMMPS Script at: {lammps_script_path}")

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
