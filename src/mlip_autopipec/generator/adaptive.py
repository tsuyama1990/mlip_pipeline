import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.generator.interface import BaseGenerator
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

    def _generate_lammps_input(self, temperature: float, steps: int) -> Path:
        """
        Generates a LAMMPS input script for MD exploration and writes it to a temporary file.

        Returns:
            Path to the generated input file.
        """
        template = self.policy.lammps_template
        content = template.format(temperature=temperature, steps=steps)

        # Create a named temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as tmp:
            tmp.write(content)
            path = Path(tmp.name)

        logger.debug(f"Generated LAMMPS input file at {path}")
        return path

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

        # Generate LAMMPS script (file-based)
        lammps_script_path = self._generate_lammps_input(temperature, self.policy.md_steps)
        # In a real scenario, we would pass this path to the Dynamics engine.
        # For now, we just log it and clean it up (or let OS handle tmp, but we set delete=False above).
        # To avoid clutter in mock mode, we delete it immediately after "use".
        try:
            logger.debug(f"LAMMPS Script generated at: {lammps_script_path}")
        finally:
            # Clean up the temp file
            if lammps_script_path.exists():
                lammps_script_path.unlink()

        count = context.get("count", self.config.mock_count)

        # For Cycle 02, we mock the MD execution by using RandomGenerator.
        # We accept the RandomGenerator output and re-label it.

        if self.random_gen:
            for s in self.random_gen.explore({"count": count}):
                s.provenance = f"md_{temperature}K"
                yield s
        else:
            msg = "AdaptiveGenerator requires a seed structure for mock execution (via RandomGenerator)."
            logger.error(msg)
            raise ValueError(msg)
