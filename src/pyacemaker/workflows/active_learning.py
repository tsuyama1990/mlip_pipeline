"""Standard Active Learning Workflow."""

from collections.abc import Callable, Iterable, Iterator
from itertools import chain
from typing import Any

from loguru import logger

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.dataset import DatasetSplitter, SeedSelector
from pyacemaker.core.interfaces import (
    CycleResult,
    DynamicsEngine,
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
from pyacemaker.oracle.dataset import DatasetManager


class StandardActiveLearningWorkflow:
    """Standard Active Learning Loop Workflow."""

    def __init__(
        self,
        config: PYACEMAKERConfig,
        dataset_manager: DatasetManager,
        structure_generator: StructureGenerator,
        oracle: Oracle,
        trainer: Trainer,
        dynamics_engine: DynamicsEngine,
        validator: Validator,
    ) -> None:
        """Initialize the workflow."""
        self.config = config
        self.logger = logger.bind(name="StandardAL")
        self.dataset_manager = dataset_manager

        self.structure_generator = structure_generator
        self.oracle = oracle
        self.trainer = trainer
        self.dynamics_engine = dynamics_engine
        self.validator = validator

        # State
        self.current_potential: Potential | None = None
        self.dataset_path = (
            self.config.project.root_dir / "data" / self.config.orchestrator.dataset_file
        )
        self.validation_path = (
            self.config.project.root_dir / "data" / self.config.orchestrator.validation_file
        )
        self.training_path = (
            self.config.project.root_dir / "data" / CONSTANTS.default_training_file
        )

        self.cycle_count = 0
        self.processed_items_count = 0

    def run(self) -> ModuleResult:
        """Run standard active learning loop."""
        # 0. Cold Start
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
                    status=CycleStatus.FAILED,
                    metrics=Metrics.model_validate({"cycles": self.cycle_count}),
                )

        self.logger.info("Pipeline completed")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"cycles": self.cycle_count}),
        )

    def run_cycle(self) -> CycleResult:
        """Execute one active learning cycle."""
        # 1. Training (Refinement)
        if not self._execute_phase(self._run_training_phase, "Training"):
            return CycleResult(
                status=CycleStatus.FAILED, metrics=Metrics(), error="Training phase failed"
            )

        # 2. Validation
        if not self._run_validation_phase():
            self.logger.error("Cycle halted due to validation failure.")
            return CycleResult(
                status=CycleStatus.FAILED, metrics=Metrics(), error="Validation failed"
            )

        # 3. Exploration (MD) & Selection
        try:
            selected_structures = self._run_exploration_and_selection_phase()
        except Exception as e:
            self.logger.exception("Exploration/Selection phase failed")
            return CycleResult(status=CycleStatus.FAILED, metrics=Metrics(), error=str(e))

        if selected_structures is None:
            return CycleResult(status=CycleStatus.CONVERGED, metrics=Metrics())

        # 5. Calculation (Oracle)
        if not self._execute_phase(
            lambda: self._run_calculation_phase(selected_structures), "Calculation"
        ):
            return CycleResult(
                status=CycleStatus.FAILED, metrics=Metrics(), error="Calculation phase failed"
            )

        return CycleResult(status=CycleStatus.TRAINING, metrics=Metrics())

    def _execute_phase(self, phase_func: Callable[[], Any], phase_name: str) -> bool:
        """Execute a phase with error handling."""
        try:
            phase_func()
        except Exception:
            self.logger.exception(f"{phase_name} phase failed")
            return False
        return True

    def _run_training_phase(self) -> None:
        """Execute training phase with incremental partitioning."""
        self.logger.info(f"Phase: Training (Incremental from index {self.processed_items_count})")

        # 1. Incremental Partitioning
        splitter = DatasetSplitter(
            self.dataset_path,
            self.validation_path,
            self.dataset_manager,
            self.config.orchestrator.validation_split,
            self.config.orchestrator.max_validation_size,
            buffer_size=self.config.orchestrator.validation_buffer_size,
            start_index=self.processed_items_count,
        )

        new_training_items = splitter.train_stream()

        atoms_iter = (metadata_to_atoms(s) for s in new_training_items)
        self.dataset_manager.save_iter(
            atoms_iter,
            self.training_path,
            mode="ab",
            calculate_checksum=False,
        )

        added_count = splitter.processed_count
        self.processed_items_count += added_count
        self.logger.info(
            f"Processed {added_count} new items. Total processed: {self.processed_items_count}"
        )

        # 2. Train on Full Training Set
        if not self.training_path.exists():
            self.logger.warning("No training data available.")
            return

        def training_stream() -> Iterator[StructureMetadata]:
            for atoms in self.dataset_manager.load_iter(self.training_path):
                yield atoms_to_metadata(atoms)

        potential = self.trainer.train(training_stream(), self.current_potential)
        self.current_potential = potential

    def _run_validation_phase(self) -> bool:
        """Execute validation phase."""
        self.logger.info("Phase: Validation")
        if not self.current_potential:
            self.logger.warning("No potential to validate.")
            return False

        if not self.validation_path.exists():
            self.logger.warning("Empty validation set.")
            return True

        def validation_stream() -> Iterator[StructureMetadata]:
            yield from (
                atoms_to_metadata(atoms)
                for atoms in self.dataset_manager.load_iter(self.validation_path)
            )

        try:
            val_result = self.validator.validate(self.current_potential, validation_stream())
        except Exception:
            self.logger.exception("Validation failed during processing")
            return False

        if val_result.status == "failed":
            self.logger.error(f"Validation failed: {val_result.metrics}")
            return False

        return True

    def _run_exploration_and_selection_phase(
        self,
    ) -> Iterable[StructureMetadata] | None:
        """Execute exploration and selection phases."""
        self.logger.info("Phase: Exploration")
        if not self.current_potential:
            return None

        selector = SeedSelector(self.dataset_manager)
        seeds = selector.get_seeds(
            self.validation_path,
            self.training_path,
            self.structure_generator,
            n_seeds=20,
        )

        if not seeds:
            self.logger.warning("No seeds available for exploration.")
            return None

        self.logger.info(f"Exploration starting with {len(seeds)} seeds.")

        high_uncertainty_stream, max_gamma_container = self._run_exploration_stream(seeds)

        if high_uncertainty_stream is None:
             self.logger.info("No high uncertainty structures found. Converged?")
             return None

        self.logger.info("Phase: Selection")
        active_set_structures = self._run_selection_phase(high_uncertainty_stream)

        self.logger.info(f"Exploration max gamma: {max_gamma_container[0]:.2f}")

        if not active_set_structures:
             self.logger.warning("ActiveSet returned no structures. Calculation skipped.")
             return []

        return active_set_structures

    def _run_exploration_stream(
        self, seeds: list[StructureMetadata]
    ) -> tuple[Iterator[StructureMetadata] | None, list[float]]:
        """Run exploration and return a monitored stream."""
        try:
            high_uncertainty_iter = self.dynamics_engine.run_exploration(
                self.current_potential, seeds  # type: ignore[arg-type]
            )

            try:
                first_structure = next(high_uncertainty_iter)
            except StopIteration:
                return None, [0.0]

            high_uncertainty_stream = chain([first_structure], high_uncertainty_iter)
            max_gamma_container = [0.0]

            def stats_spy(iterator: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]:
                try:
                    for s in iterator:
                        if s.uncertainty_state and s.uncertainty_state.gamma_max:
                            max_gamma_container[0] = max(
                                max_gamma_container[0], s.uncertainty_state.gamma_max
                            )
                        yield s
                except Exception:
                    self.logger.exception("Error during exploration streaming")
                    return

            return stats_spy(high_uncertainty_stream), max_gamma_container

        except Exception:
            self.logger.exception("Failed to initialize exploration stream")
            return None, [0.0]

    def _run_selection_phase(
        self, high_uncertainty_stream: Iterator[StructureMetadata]
    ) -> Iterable[StructureMetadata] | None:
        """Execute selection phase on the exploration stream."""
        n_local = self.config.orchestrator.n_local_candidates

        candidates_iter = self.structure_generator.generate_batch_candidates(
            high_uncertainty_stream,
            n_candidates_per_seed=n_local,
            cycle=self.cycle_count,
        )

        n_select = self.config.orchestrator.n_active_set_select
        active_set = self.trainer.select_active_set(candidates_iter, n_select=n_select)

        if active_set.dataset_path and active_set.dataset_path.exists():
            def structure_loader() -> Iterator[StructureMetadata]:
                if active_set.dataset_path:
                    for atoms in self.dataset_manager.load_iter(active_set.dataset_path):
                        yield atoms_to_metadata(atoms)

            return structure_loader()

        if active_set.structures:
            return active_set.structures

        return None

    def _run_calculation_phase(self, structures: Iterable[StructureMetadata]) -> None:
        """Execute calculation phase."""
        self.logger.info("Phase: Calculation (Streaming)")
        new_data = self.oracle.compute_batch(structures)
        self._save_dataset_stream(new_data)

    def _save_dataset_stream(self, stream: Iterator[StructureMetadata]) -> None:
        """Convert metadata stream to atoms and save to dataset."""
        atoms_stream = (metadata_to_atoms(s) for s in stream)
        self.dataset_manager.save_iter(
            atoms_stream, self.dataset_path, mode="ab", calculate_checksum=False
        )
        checksum_path = self.dataset_path.with_suffix(self.dataset_path.suffix + ".sha256")
        if checksum_path.exists():
            try:
                checksum_path.unlink()
            except OSError:
                self.logger.warning("Failed to remove stale checksum file.")

    def _run_cold_start(self) -> None:
        """Execute cold start to generate initial dataset."""
        initial_structures = self.structure_generator.generate_initial_structures()
        computed_stream = self.oracle.compute_batch(initial_structures)
        self._save_dataset_stream(computed_stream)
