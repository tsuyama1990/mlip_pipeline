import logging
from collections.abc import Iterator
from typing import Any

from ase import Atoms

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

    def _generate_lammps_input(self, temperature: float, steps: int) -> str:
        """Generates a simple LAMMPS input string for MD exploration."""
        return f"""
units metal
atom_style atomic
boundary p p p

# Potential setup (placeholder)
pair_style none

# MD Settings
velocity all create {temperature} 12345 dist gaussian
fix 1 all nvt temp {temperature} {temperature} 0.1
timestep 0.001

run {steps}
"""

    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        cycle = context.get("cycle", 0)

        # Temperature Schedule
        schedule = self.policy.temperature_schedule
        if not schedule:
            schedule = [300.0]

        # Pick temperature based on cycle (cycle 0 -> index 0, etc.)
        # If cycle >= len, clamp to last
        idx = min(cycle, len(schedule) - 1)
        temperature = schedule[idx]

        logger.info(f"AdaptiveGenerator: Cycle {cycle}, using Temperature={temperature}K")

        # Log the LAMMPS script for verification/debug
        lammps_script = self._generate_lammps_input(temperature, self.policy.md_steps)
        logger.debug(f"Generated LAMMPS Script:\n{lammps_script}")

        count = context.get("count", self.config.mock_count)

        # For Cycle 02, we mock the MD execution by using RandomGenerator or creating dummies
        # but labeling them with the MD provenance.

        if self.random_gen:
            # Use RandomGenerator to produce perturbed structures
            # Ideally we would scale perturbation with temperature, but RandomGenerator uses fixed config.
            # We accept the RandomGenerator output and re-label it.
            for s in self.random_gen.explore({"count": count}):
                s.provenance = f"md_{temperature}K"
                yield s
        else:
            # Fallback if no seed
            logger.info("No seed structure for RandomGenerator, generating dummy structures.")
            for _ in range(count):
                atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
                yield Structure(atoms=atoms, provenance=f"md_{temperature}K", label_status="unlabeled")
