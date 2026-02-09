import logging
from collections.abc import Iterator
from typing import Any

from mlip_autopipec.components.generator.base import BaseGenerator
from mlip_autopipec.components.generator.builder import (
    BulkBuilder,
    StructureBuilder,
    SurfaceBuilder,
)
from mlip_autopipec.components.generator.policy import ExplorationPolicy
from mlip_autopipec.components.generator.rattle import RattleTransform, StrainTransform
from mlip_autopipec.domain_models.config import AdaptiveGeneratorConfig
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class AdaptiveGenerator(BaseGenerator):
    """
    Adaptive Structure Generator (Pragmatic Implementation).

    Architecture Decision:
    This implementation focuses on a pragmatic "Placeholder" strategy for initial system stability.
    It uses a simplified `ExplorationPolicy` that switches between Bulk and Surface generation
    based on configured ratios.

    In the Batch Active Learning architecture:
    - Cycle 0: This Generator creates initial structures (Cold Start).
    - Cycle N: This Generator provides seed structures for Dynamics exploration.
    """

    def __init__(self, config: AdaptiveGeneratorConfig) -> None:
        super().__init__(config)
        self.policy = ExplorationPolicy()
        self.builders: dict[str, StructureBuilder] = {
            "bulk": BulkBuilder(),
            "surface": SurfaceBuilder(),
        }

    def generate(
        self, n_structures: int, cycle: int = 0, metrics: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        """
        Generate structures using configured policy.

        Args:
            n_structures: Total number of structures to generate.
            cycle: The current active learning cycle number.
            metrics: Optional metrics from the previous cycle.

        Yields:
            Generated Structure objects.
        """
        logger.info(f"Generating {n_structures} structures using AdaptiveGenerator (Cycle {cycle})")

        cfg = self.config
        if not isinstance(cfg, AdaptiveGeneratorConfig):
            msg = f"Invalid config type for AdaptiveGenerator: {type(cfg)}"
            raise TypeError(msg)

        # Instantiate Transforms with config
        rattle = RattleTransform(stdev=cfg.rattle_strength)
        strain = StrainTransform(strain_range=cfg.strain_range)

        # Determine ratios
        # Priority: 1. Schedule in policy.ratio_schedule for current cycle
        #           2. Default in policy.ratio_schedule
        #           3. Legacy cfg.policy_ratios

        ratios = cfg.policy_ratios  # Default legacy
        cycle_key = f"cycle_{cycle}"

        if cfg.policy.ratio_schedule:
            if cycle_key in cfg.policy.ratio_schedule:
                ratios = cfg.policy.ratio_schedule[cycle_key]
                logger.info(f"Using scheduled ratios for cycle {cycle}: {ratios}")
            elif "default" in cfg.policy.ratio_schedule:
                ratios = cfg.policy.ratio_schedule["default"]

        # Get Tasks from Policy
        tasks = self.policy.decide_next_batch(
            current_cycle=cycle,
            current_metrics=metrics or {},
            n_total=n_structures,
            ratios=ratios,
        )

        # Execute Tasks
        generated_count = 0
        for task in tasks:
            builder = self.builders.get(task.builder_name)
            if not builder:
                logger.warning(f"Unknown builder type: {task.builder_name}. Skipping.")
                continue

            structures = builder.build(task.n_structures, cfg)

            for s in structures:
                current_structure = s
                # Apply transforms based on type
                struct_type = current_structure.tags.get("type")

                # Apply strain mainly to bulk to explore lattice parameters
                if struct_type == "bulk":
                    current_structure = strain.apply(current_structure)

                # Apply rattle to everything to break perfect symmetry
                current_structure = rattle.apply(current_structure)

                yield current_structure
                generated_count += 1

        if generated_count < n_structures:
            logger.warning(
                f"Requested {n_structures} but generated {generated_count}. "
                "Check policy or builders."
            )

    def enhance(self, structure: Structure) -> Iterator[Structure]:
        """
        Enhance a structure by generating local candidates (e.g. for halted structures).
        Uses RattleTransform to generate perturbed candidates.
        """
        cfg = self.config
        if not isinstance(cfg, AdaptiveGeneratorConfig):
            # Fallback to just returning structure if config mismatch (unlikely)
            yield structure
            return

        # Always yield the anchor structure first
        yield structure

        n_candidates = cfg.policy.n_candidates
        rattle = RattleTransform(stdev=cfg.rattle_strength)

        for _ in range(n_candidates):
            candidate = structure.model_deep_copy()
            candidate = rattle.apply(candidate)

            # Update tags to trace lineage
            candidate.tags["provenance"] = "local_candidate"
            candidate.tags["parent_halt"] = structure.tags.get("provenance", "unknown")

            # Reset labels/uncertainty
            candidate.energy = None
            candidate.forces = None
            candidate.stress = None
            candidate.uncertainty = None

            yield candidate

    def __repr__(self) -> str:
        return f"<AdaptiveGenerator(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"AdaptiveGenerator({self.name})"
