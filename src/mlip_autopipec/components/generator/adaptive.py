import logging
from collections.abc import Iterator

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

    def generate(self, n_structures: int) -> Iterator[Structure]:
        """
        Generate structures using configured policy.

        Args:
            n_structures: Total number of structures to generate.

        Yields:
            Generated Structure objects.
        """
        logger.info(f"Generating {n_structures} structures using AdaptiveGenerator")

        cfg = self.config
        if not isinstance(cfg, AdaptiveGeneratorConfig):
            msg = f"Invalid config type for AdaptiveGenerator: {type(cfg)}"
            raise TypeError(msg)

        # Instantiate Transforms with config
        rattle = RattleTransform(stdev=cfg.rattle_strength)
        strain = StrainTransform(strain_range=cfg.strain_range)

        # Get Tasks from Policy
        # We pass current_cycle=0 to treat this as initial generation or simple seeding.
        # Future work: expose method to update policy state if needed.
        tasks = self.policy.decide_next_batch(
            current_cycle=0,
            current_metrics={},
            n_total=n_structures,
            ratios=cfg.policy_ratios,
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

    def __repr__(self) -> str:
        return f"<AdaptiveGenerator(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"AdaptiveGenerator({self.name})"
