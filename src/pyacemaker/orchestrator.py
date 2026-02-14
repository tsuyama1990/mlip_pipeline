"""Orchestrator module implementation."""

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TypeVar

from pyacemaker.core.base import BaseModule, Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    IOrchestrator,
    Oracle,
    StructureGenerator,
    Trainer,
    Validator,
)
from pyacemaker.domain_models.models import (
    CycleStatus,
    Potential,
    StructureMetadata,
)
from pyacemaker.modules.dynamics_engine import LAMMPSEngine
from pyacemaker.modules.oracle import MockOracle
from pyacemaker.modules.structure_generator import (
    AdaptiveStructureGenerator,
    RandomStructureGenerator,
)
from pyacemaker.modules.trainer import PacemakerTrainer
from pyacemaker.modules.validator import MockValidator

T = TypeVar("T", bound=BaseModule)


def _create_default_module(module_class: type[T], config: PYACEMAKERConfig) -> T:
    """Factory to create a default module instance."""
    return module_class(config)


class Orchestrator(IOrchestrator):
    """Main Orchestrator for the active learning cycle."""

    def __init__(
        self,
        config: PYACEMAKERConfig,
        structure_generator: StructureGenerator | None = None,
        oracle: Oracle | None = None,
        trainer: Trainer | None = None,
        dynamics_engine: DynamicsEngine | None = None,
        validator: Validator | None = None,
    ) -> None:
        """Initialize the orchestrator and sub-modules.

        Dependencies can be injected; otherwise, default implementations are instantiated via factory.
        """
        super().__init__(config)
        self.config = config

        # Dependency Injection with fallbacks using factory pattern
        if structure_generator:
            self.structure_generator = structure_generator
        elif config.structure_generator.strategy == "adaptive":
            self.structure_generator = AdaptiveStructureGenerator(config)
        else:
            self.structure_generator = RandomStructureGenerator(config)

        self.oracle: Oracle = oracle or _create_default_module(MockOracle, config)
        self.trainer: Trainer = trainer or _create_default_module(PacemakerTrainer, config)
        self.dynamics_engine: DynamicsEngine = dynamics_engine or _create_default_module(
            LAMMPSEngine, config
        )
        self.validator: Validator = validator or _create_default_module(MockValidator, config)

        # State
        self.current_potential: Potential | None = None
        self.dataset: list[StructureMetadata] = []
        self.cycle_count = 0

    def run(self) -> ModuleResult:
        """Run the full active learning pipeline."""
        self.logger.info("Starting Active Learning Pipeline")

        # 0. Cold Start (Initial Dataset)
        if not self.dataset:
            self.logger.info("Cold Start: Generating initial structures")
            initial_structures = self.structure_generator.generate_initial_structures()
            # Stream: initial_structures (iter) -> compute_batch (iter) -> extend (consumes)
            # This avoids holding intermediate lists in memory.
            self.dataset.extend(self.oracle.compute_batch(initial_structures))

        # Main Loop
        max_cycles = self.config.orchestrator.max_cycles
        while self.cycle_count < max_cycles:
            self.cycle_count += 1
            self.logger.info(f"--- Cycle {self.cycle_count}/{max_cycles} ---")

            status = self.run_cycle()

            if status == CycleStatus.CONVERGED:
                self.logger.info("Convergence reached!")
                break
            if status == CycleStatus.FAILED:
                self.logger.error("Cycle failed!")
                return ModuleResult(
                    status="failed",
                    metrics=Metrics.model_validate({"cycles": self.cycle_count}),
                )

        self.logger.info("Pipeline completed")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate(
                {"cycles": self.cycle_count, "dataset_size": len(self.dataset)}
            ),
        )

    def run_cycle(self) -> CycleStatus:
        """Execute one active learning cycle."""
        # 1. Training (Refinement)
        self.logger.info("Phase: Training")
        potential = self.trainer.train(self.dataset, self.current_potential)
        self.current_potential = potential

        # 2. Validation
        self.logger.info("Phase: Validation")
        # Split dataset for validation (simple holdout for now)
        test_size = max(1, int(len(self.dataset) * 0.1))
        # Use slicing carefully, dataset is a list so this creates a copy
        # For very large datasets, we might need indices instead
        test_set = self.dataset[:test_size]
        val_result = self.validator.validate(potential, test_set)

        if val_result.status == "failed":
            self.logger.warning("Validation failed, but continuing for exploration...")

        # 3. Exploration (MD)
        self.logger.info("Phase: Exploration")
        high_uncertainty_iter = self.dynamics_engine.run_exploration(potential)

        # Check if we found anything without consuming the whole stream
        try:
            first_structure = next(high_uncertainty_iter)
        except StopIteration:
            self.logger.info("No high uncertainty structures found. Converged?")
            return CycleStatus.CONVERGED

        # Reconstruct iterator
        high_uncertainty_stream = chain([first_structure], high_uncertainty_iter)

        # Spy on the stream to calculate metrics (max gamma) without materializing list
        max_gamma = 0.0

        def stats_spy(iterator: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
            nonlocal max_gamma
            for s in iterator:
                if s.uncertainty_state and s.uncertainty_state.gamma_max:
                    max_gamma = max(max_gamma, s.uncertainty_state.gamma_max)
                yield s

        high_uncertainty_stream_with_stats = stats_spy(high_uncertainty_stream)

        # 4. Selection (Local Candidates & Active Set)
        self.logger.info("Phase: Selection")
        n_local = self.config.orchestrator.n_local_candidates

        # Stream candidates generation
        candidates_iter = self.structure_generator.generate_batch_candidates(
            high_uncertainty_stream_with_stats, n_candidates_per_seed=n_local
        )

        # Pass iterator directly to Trainer to avoid materializing large lists in memory.
        # The Trainer streams candidates to a temporary file for `pace_activeset`.
        n_select = self.config.orchestrator.n_active_set_select
        active_set = self.trainer.select_active_set(candidates_iter, n_select=n_select)

        # Log stats after consumption (candidates_iter is consumed by select_active_set)
        self.logger.info(f"Exploration max gamma: {max_gamma:.2f}")

        # Retrieve selected structures directly from ActiveSet metadata
        if active_set.structures:
            selected_structures = active_set.structures
        else:
            self.logger.warning("ActiveSet returned no structure objects. Calculation skipped.")
            selected_structures = []

        # 5. Calculation (Oracle)
        self.logger.info(f"Phase: Calculation ({len(selected_structures)} structures)")
        # Stream processing: list -> compute_batch(iter) -> extend
        new_data = self.oracle.compute_batch(selected_structures)
        self.dataset.extend(new_data)

        return CycleStatus.TRAINING
