"""Orchestrator module implementation."""

import secrets
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TypeVar

from pyacemaker.core.base import BaseModule, Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    CycleResult,
    DynamicsEngine,
    IOrchestrator,
    Oracle,
    StructureGenerator,
    Trainer,
    Validator,
)
from pyacemaker.core.utils import atoms_to_metadata, metadata_to_atoms
from pyacemaker.domain_models.models import (
    CycleStatus,
    Potential,
    StructureMetadata,
)
from pyacemaker.modules.dynamics_engine import LAMMPSEngine
from pyacemaker.modules.oracle import DFTOracle, MockOracle
from pyacemaker.modules.structure_generator import (
    AdaptiveStructureGenerator,
    RandomStructureGenerator,
)
from pyacemaker.modules.trainer import PacemakerTrainer
from pyacemaker.modules.validator import MockValidator
from pyacemaker.oracle.dataset import DatasetManager

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

        # Select Oracle implementation based on config
        oracle_cls = MockOracle if config.oracle.mock else DFTOracle
        self.oracle: Oracle = oracle or _create_default_module(oracle_cls, config)

        self.trainer: Trainer = trainer or _create_default_module(PacemakerTrainer, config)
        self.dynamics_engine: DynamicsEngine = dynamics_engine or _create_default_module(
            LAMMPSEngine, config
        )
        self.validator: Validator = validator or _create_default_module(MockValidator, config)

        # State
        self.current_potential: Potential | None = None
        # Dataset is now file-based to prevent OOM
        # Use config option instead of hardcoded
        self.dataset_path = (
            self.config.project.root_dir / "data" / self.config.orchestrator.dataset_file
        )
        self.dataset_manager = DatasetManager()
        self.cycle_count = 0

    def run(self) -> ModuleResult:
        """Run the full active learning pipeline."""
        self.logger.info("Starting Active Learning Pipeline")

        # 0. Cold Start (Initial Dataset)
        if not self.dataset_path.exists():
            self.logger.info("Cold Start: Generating initial structures")
            self._run_cold_start()

        # Main Loop
        max_cycles = self.config.orchestrator.max_cycles
        while self.cycle_count < max_cycles:
            self.cycle_count += 1
            self.logger.info(f"--- Cycle {self.cycle_count}/{max_cycles} ---")

            result = self.run_cycle()

            if result.status == CycleStatus.CONVERGED:
                self.logger.info("Convergence reached!")
                break
            if result.status == CycleStatus.FAILED:
                self.logger.error(f"Cycle failed! Reason: {result.error}")
                return ModuleResult(
                    status="failed",
                    metrics=Metrics.model_validate({"cycles": self.cycle_count}),
                )

        self.logger.info("Pipeline completed")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"cycles": self.cycle_count}),
        )

    def _run_cold_start(self) -> None:
        """Execute cold start to generate initial dataset."""
        initial_structures = self.structure_generator.generate_initial_structures()
        # Stream: initial_structures (iter) -> compute_batch (iter) -> save (append)
        computed_stream = self.oracle.compute_batch(initial_structures)

        # Optimization: Don't convert back and forth if not needed,
        # but DatasetManager expects Atoms. And compute_batch returns StructureMetadata.
        # So conversion IS needed unless we change interfaces.
        # But we can do it lazily.
        atoms_stream = (metadata_to_atoms(s) for s in computed_stream)
        self.dataset_manager.save_iter(atoms_stream, self.dataset_path, mode="ab")

    def run_cycle(self) -> CycleResult:
        """Execute one active learning cycle."""
        # 1. Training (Refinement)
        try:
            self._run_training_phase()
        except Exception as e:
            self.logger.exception("Training phase failed")
            return CycleResult(status=CycleStatus.FAILED, metrics=Metrics(), error=str(e))

        # 2. Validation
        if not self._run_validation_phase():
            self.logger.error("Cycle halted due to validation failure.")
            return CycleResult(
                status=CycleStatus.FAILED, metrics=Metrics(), error="Validation failed"
            )

        # 3. Exploration (MD) & Selection
        # Exploration returns high uncertainty structures
        # Selection filters them and generates candidates
        try:
            selected_structures = self._run_exploration_and_selection_phase()
        except Exception as e:
            self.logger.exception("Exploration/Selection phase failed")
            return CycleResult(status=CycleStatus.FAILED, metrics=Metrics(), error=str(e))

        if selected_structures is None:
            return CycleResult(status=CycleStatus.CONVERGED, metrics=Metrics())

        # 5. Calculation (Oracle)
        try:
            self._run_calculation_phase(selected_structures)
        except Exception as e:
            self.logger.exception("Calculation phase failed")
            return CycleResult(status=CycleStatus.FAILED, metrics=Metrics(), error=str(e))

        return CycleResult(status=CycleStatus.TRAINING, metrics=Metrics())

    def _run_training_phase(self) -> None:
        """Execute training phase with file-based streaming."""
        self.logger.info("Phase: Training")

        # Get stream and list reference (list populated during iteration of stream)
        stream, val_list = self._split_dataset_streams()

        # Training consumes the stream, side-effect populates val_list
        potential = self.trainer.train(stream, self.current_potential)

        self.current_potential = potential
        self._validation_set = val_list

    def _split_dataset_streams(self) -> tuple[Iterator[StructureMetadata], list[StructureMetadata]]:
        """Split dataset into training stream and validation list (Probabilistic)."""
        split_ratio = self.config.orchestrator.validation_split
        max_val = self.config.orchestrator.max_validation_size
        val_list: list[StructureMetadata] = []

        def train_stream() -> Iterator[StructureMetadata]:
            if not self.dataset_path.exists():
                return

            # Use DatasetManager to load items
            for atoms in self.dataset_manager.load_iter(self.dataset_path):
                # Probabilistic split: if random < split_ratio AND we are under cap
                # If split_ratio is 1.0, we want everything in validation, UP TO max_val.
                # But train_stream must yield training data. If everything goes to val, train is empty?
                # That would break training.
                # Assuming standard split logic:
                is_full = len(val_list) >= max_val
                should_validate = (not is_full) and (secrets.SystemRandom().random() < split_ratio)

                if should_validate:
                    val_list.append(atoms_to_metadata(atoms))
                else:
                    yield atoms_to_metadata(atoms)

        return train_stream(), val_list

    def _run_validation_phase(self) -> bool:
        """Execute validation phase.

        Returns:
            bool: True if validation passed, False otherwise.
        """
        self.logger.info("Phase: Validation")
        if not self.current_potential:
            self.logger.warning("No potential to validate.")
            return False

        test_set = getattr(self, "_validation_set", [])
        if not test_set:
            self.logger.warning("Empty validation set.")
            # We rely on training phase populate
            return True

        val_result = self.validator.validate(self.current_potential, test_set)

        if val_result.status == "failed":
            self.logger.error(f"Validation failed: {val_result.metrics}")
            return False

        return True

    def _run_exploration_and_selection_phase(self) -> list[StructureMetadata] | None:
        """Execute exploration and selection phases.

        Returns:
            List of selected structures for calculation, or None if converged.
        """
        self.logger.info("Phase: Exploration")
        if not self.current_potential:
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
            high_uncertainty_stream_with_stats,
            n_candidates_per_seed=n_local,
            cycle=self.cycle_count,
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
        # Stream processing: list -> compute_batch(iter) -> save (append)
        new_data = self.oracle.compute_batch(structures)

        atoms_stream = (metadata_to_atoms(s) for s in new_data)
        self.dataset_manager.save_iter(atoms_stream, self.dataset_path, mode="ab")
