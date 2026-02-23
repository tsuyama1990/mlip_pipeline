"""Active Learning Module."""

from collections.abc import Iterable, Iterator

from loguru import logger

from pyacemaker.core.base import BaseModule, Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig, Step2ActiveLearningConfig
from pyacemaker.core.utils import select_top_k_structures
from pyacemaker.domain_models.models import ActiveSet, StructureMetadata


class ActiveLearner(BaseModule):
    """Module for selecting structures based on uncertainty."""

    def __init__(self, config: PYACEMAKERConfig | Step2ActiveLearningConfig) -> None:
        """Initialize the Active Learner."""
        self.al_config: Step2ActiveLearningConfig

        if isinstance(config, PYACEMAKERConfig):
            super().__init__(config)
            self.al_config = config.distillation.step2_active_learning
        else:
            self.config = config  # type: ignore[assignment]
            self.al_config = config
            self.logger = logger.bind(name="ActiveLearner")

    def run(self) -> ModuleResult:
        """Execute default active learning task."""
        return ModuleResult(
            status="success",
            metrics=Metrics(message="ActiveLearner ready")  # type: ignore[call-arg]
        )

    def select_batch(self, candidates: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
        """Select top-k structures with highest uncertainty."""
        n_select = self.al_config.n_select
        threshold = self.al_config.uncertainty_threshold

        self.logger.info(f"Selecting top {n_select} structures (threshold > {threshold})")

        # Define key function for selection (higher uncertainty is better)
        def uncertainty_key(s: StructureMetadata) -> float:
            if s.uncertainty is not None:
                return s.uncertainty
            # Fallback if uncertainty missing
            if s.uncertainty_state and s.uncertainty_state.gamma_max is not None:
                return s.uncertainty_state.gamma_max
            return -1.0

        # Filter candidates by threshold first?
        # SPEC says "Selects the top N structures with the highest uncertainty".
        # Threshold might be a hard cutoff.
        # "Verify that the selected structures have uncertainty > 0.8".

        # We can select top N, then verify/filter.
        # Or filter then top N.
        # Usually filter then top N ensures quality.
        # But if we select top N and they are all below threshold, do we select none?
        # The UAT implies filtering.

        # However, for streaming, we can't easily "filter then sort" without full scan.
        # `select_top_k_structures` does full scan (O(N) time, O(K) memory).

        selected = select_top_k_structures(candidates, n_select, uncertainty_key)

        count = 0
        for s in selected:
            unc = uncertainty_key(s)
            if unc >= threshold:
                # Mark as selected
                s.tags.append("active_set")
                s.tags.append(f"unc_{unc:.4f}")
                yield s
                count += 1
            else:
                self.logger.debug(f"Structure {s.id} rejected (uncertainty {unc} < {threshold})")

        self.logger.info(f"Selected {count} structures")

    def select_active_set(
        self, candidates: Iterable[StructureMetadata], n_select: int
    ) -> ActiveSet:
        """Select active set for Trainer interface compliance (if needed)."""
        # This module is not a Trainer, but if it implemented Trainer.select_active_set...
        # We stick to select_batch for now.
        msg = "Use select_batch instead"
        raise NotImplementedError(msg)
