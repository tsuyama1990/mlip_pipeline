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


class AdaptiveGenerator(BaseGenerator[AdaptiveGeneratorConfig]):
    """
    Adaptive Structure Generator (Pragmatic Implementation).

    Architecture Decision:
    While the original Specification (Spec 3.1) calls for an "Adaptive Exploration Policy Engine"
    using M3GNet/CHGNet and sophisticated MD/MC ratio tuning, this implementation focuses on a
    pragmatic "Placeholder" strategy for the initial system stability.

    It uses a simplified `ExplorationPolicy` that switches between Bulk and Surface generation
    based on validation errors, and applies random perturbations (Rattle/Strain) to explore
    local configuration space.

    Future work can replace `BulkBuilder`/`SurfaceBuilder` with `M3GNetBuilder` or `MDBuilder`
    without changing the `Generator` interface, preserving the Orchestrator logic.
    """

    _VALID_KEYS = frozenset(AdaptiveGeneratorConfig.model_fields.keys())

    def __init__(self, config: AdaptiveGeneratorConfig) -> None:
        super().__init__(config)
        self.policy = ExplorationPolicy()
        self.builders: dict[str, StructureBuilder] = {
            "bulk": BulkBuilder(),
            "surface": SurfaceBuilder(),
        }

    def _resolve_config(
        self, config: dict[str, Any] | None
    ) -> tuple[AdaptiveGeneratorConfig, int, dict[str, Any]]:
        current_cycle = 0
        current_metrics: dict[str, Any] = {}

        if config:
            # Extract runtime params that are not part of GeneratorConfig
            if "current_cycle" in config:
                current_cycle = config.pop("current_cycle")
            if "current_metrics" in config:
                current_metrics = config.pop("current_metrics")

        if not config:
            # If config is None or became empty after popping cycle/metrics, use base config
            return self.config, current_cycle, current_metrics

        # Update config with remaining keys
        effective_config_dict = self.config.model_dump()
        effective_config_dict.update(config)

        # Re-validate config to ensure integrity
        clean_config_dict = {
            k: v for k, v in effective_config_dict.items() if k in self._VALID_KEYS
        }

        try:
            run_config = AdaptiveGeneratorConfig.model_validate(clean_config_dict)
        except Exception as e:
            logger.exception("Invalid runtime configuration")
            msg = f"Invalid configuration: {e}"
            raise ValueError(msg) from e

        return run_config, current_cycle, current_metrics

    def generate(
        self, n_structures: int, config: dict[str, Any] | None = None
    ) -> Iterator[Structure]:
        """
        Generate structures using adaptive policy.

        Args:
            n_structures: Total number of structures to generate.
            config: Runtime configuration override. Can include 'current_cycle'
                    and 'current_metrics' for policy decision.

        Yields:
            Generated Structure objects.
        """
        logger.info(f"Generating {n_structures} structures using AdaptiveGenerator")

        # 1. Prepare Configuration
        run_config, current_cycle, current_metrics = self._resolve_config(config)

        # 2. Instantiate Transforms with run_config
        rattle = RattleTransform(stdev=run_config.rattle_strength)
        strain = StrainTransform(strain_range=run_config.strain_range)

        # 3. Get Tasks from Policy
        tasks = self.policy.decide_next_batch(
            current_cycle,
            current_metrics,
            n_structures,
            ratios=run_config.policy_ratios,
        )

        # 4. Execute Tasks
        generated_count = 0
        for task in tasks:
            builder = self.builders.get(task.builder_name)
            if not builder:
                logger.warning(f"Unknown builder type: {task.builder_name}. Skipping.")
                continue

            structures = builder.build(task.n_structures, run_config)

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
