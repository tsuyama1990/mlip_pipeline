"""Orchestrator module implementation."""

from collections.abc import Iterable, Iterator
from itertools import chain, islice
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
        # Logic for selecting structure generator strategy
        if structure_generator:
            self.structure_generator = structure_generator
        else:
            # Factory logic for structure generator based on config
            sg_cls = (
                AdaptiveStructureGenerator
                if config.structure_generator.strategy == "adaptive"
                else RandomStructureGenerator
            )
            self.structure_generator = _create_default_module(sg_cls, config)

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
            self._run_cold_start()

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

    def _run_cold_start(self) -> None:
        """Execute cold start to generate initial dataset."""
        initial_structures = self.structure_generator.generate_initial_structures()
        # Stream: initial_structures (iter) -> compute_batch (iter) -> extend (consumes)
        # This avoids holding intermediate lists in memory before extending dataset.
        computed_stream = self.oracle.compute_batch(initial_structures)
        self.dataset.extend(computed_stream)

    def run_cycle(self) -> CycleStatus:
        """Execute one active learning cycle."""
        # 1. Training (Refinement)
        self._run_training_phase()

        # 2. Validation
        self._run_validation_phase()

        # 3. Exploration (MD) & Selection
        # Exploration returns high uncertainty structures
        # Selection filters them and generates candidates
        selected_structures = self._run_exploration_and_selection_phase()

        if selected_structures is None:
            return CycleStatus.CONVERGED

        # 5. Calculation (Oracle)
        self._run_calculation_phase(selected_structures)

        return CycleStatus.TRAINING

    def _run_training_phase(self) -> None:
        """Execute training phase."""
        self.logger.info("Phase: Training")
        potential = self.trainer.train(self.dataset, self.current_potential)
        self.current_potential = potential

    def _run_validation_phase(self) -> None:
        """Execute validation phase."""
        self.logger.info("Phase: Validation")
        if not self.current_potential:
            self.logger.warning("No potential to validate.")
            return

        # Split dataset for validation using islice to avoid copying list slice
        # Limit validation set size to prevent OOM
        test_size = min(max(1, int(len(self.dataset) * 0.1)), 1000)
        # Use islice to get an iterator over the first test_size elements
        # Note: validate method takes a list, so we must materialize this small subset.
        # But we avoid creating the full slice copy if list implementation is smart (CPython copies anyway for slice)
        # We explicitly bounded test_size to 1000 to prevent OOM when materializing this list.
        test_set = list(islice(self.dataset, test_size))
        val_result = self.validator.validate(self.current_potential, test_set)

        if val_result.status == "failed":
            # For strict mode, we should raise an error here.
            # Currently we log a warning as per original design, but this is a critical gate.
            self.logger.warning("Validation failed, but continuing for exploration...")

    def _run_exploration_and_selection_phase(self) -> list[StructureMetadata] | None:
        """Execute exploration and selection phases.

        Returns:
            List of selected structures for calculation, or None if converged.
        """
        self.logger.info("Phase: Exploration")
        if not self.current_potential:
             # Should not happen if training succeeded, but for safety
             return None

        high_uncertainty_iter = self.dynamics_engine.run_exploration(self.current_potential)

        # Check if we found anything without consuming the whole stream
        try:
            first_structure = next(high_uncertainty_iter)
        except StopIteration:
            self.logger.info("No high uncertainty structures found. Converged?")
            return None

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

        # Selection (Local Candidates & Active Set)
        self.logger.info("Phase: Selection")
        n_local = self.config.orchestrator.n_local_candidates

        # Stream candidates generation
        candidates_iter = self.structure_generator.generate_batch_candidates(
            high_uncertainty_stream_with_stats, n_candidates_per_seed=n_local
        )

        n_select = self.config.orchestrator.n_active_set_select
        active_set = self.trainer.select_active_set(candidates_iter, n_select=n_select)

        # Log stats after consumption
        self.logger.info(f"Exploration max gamma: {max_gamma:.2f}")

        if active_set.structures:
            return active_set.structures

        self.logger.warning("ActiveSet returned no structure objects. Calculation skipped.")
        return []

    def _run_calculation_phase(self, structures: list[StructureMetadata]) -> None:
        """Execute calculation phase."""
        self.logger.info(f"Phase: Calculation ({len(structures)} structures)")
        # Stream processing: list -> compute_batch(iter) -> extend
        new_data = self.oracle.compute_batch(structures)
        self.dataset.extend(new_data)
