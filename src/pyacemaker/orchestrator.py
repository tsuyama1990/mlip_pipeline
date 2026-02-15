"""Orchestrator module implementation."""

import secrets
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import TypeVar

from pyacemaker.core.base import BaseModule, Metrics, ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    CycleResult,
    DynamicsEngine,
    IOrchestrator,
    Oracle,
    StructureGenerator,
    Trainer,
)
from pyacemaker.core.interfaces import (
    Validator as ValidatorInterface,
)
from pyacemaker.core.utils import atoms_to_metadata, metadata_to_atoms
from pyacemaker.domain_models.models import (
    CycleStatus,
    Potential,
    StructureMetadata,
)
from pyacemaker.modules.dynamics_engine import EONEngine, LAMMPSEngine
from pyacemaker.modules.oracle import DFTOracle, MockOracle
from pyacemaker.modules.structure_generator import (
    AdaptiveStructureGenerator,
    RandomStructureGenerator,
)
from pyacemaker.modules.trainer import PacemakerTrainer
from pyacemaker.modules.validator import Validator
from pyacemaker.oracle.dataset import DatasetManager

T = TypeVar("T", bound=BaseModule)


def _create_default_module(module_class: type[T], config: PYACEMAKERConfig) -> T:
    """Factory to create a default module instance."""
    return module_class(config)


class DatasetSplitter:
    """Helper to split dataset into training stream and validation file."""

    def __init__(
        self,
        dataset_path: "Path",
        validation_path: "Path",
        dataset_manager: "DatasetManager",
        validation_split: float,
        max_validation_size: int,
        buffer_size: int = 100,  # Fallback, usually overridden by config
        start_index: int = 0,
    ) -> None:
        from pyacemaker.core.config import CONSTANTS

        if buffer_size == 100:  # If default
             buffer_size = CONSTANTS.default_validation_buffer_size
        self.dataset_path = dataset_path
        self.validation_path = validation_path
        self.dataset_manager = dataset_manager
        self.validation_split = validation_split
        self.max_validation_size = max_validation_size
        self.buffer_size = buffer_size
        self._rng = secrets.SystemRandom()
        # Note: We don't know current validation size without reading file or tracking state.
        # Ideally Orchestrator tracks this. For now, we assume simple appending logic.
        # The audit requirement is "incremental", so we just split NEW items.
        self._val_count = 0
        self.start_index = start_index
        self.processed_count = 0

    def train_stream(self) -> Iterator[StructureMetadata]:
        """Yield training items and save validation items to file as side effect."""
        if not self.dataset_path.exists():
            return

        val_buffer: list[StructureMetadata] = []

        # Optimization: Use start_index in load_iter to skip deserialization of old items
        stream = self.dataset_manager.load_iter(self.dataset_path, start_index=self.start_index)

        for atoms in stream:
            self.processed_count += 1
            # Simple random split for new items
            # We check max_validation_size relative to *added* items in this session for safety,
            # though ideally we'd check global size.
            is_full = self._val_count >= self.max_validation_size
            should_validate = (not is_full) and (self._rng.random() < self.validation_split)

            if should_validate:
                val_buffer.append(atoms_to_metadata(atoms))
                self._val_count += 1
                if len(val_buffer) >= self.buffer_size:
                     self._flush_validation(val_buffer)
                     val_buffer = []
            else:
                yield atoms_to_metadata(atoms)

        if val_buffer:
            self._flush_validation(val_buffer)

    def _flush_validation(self, items: list[StructureMetadata]) -> None:
        """Flush validation buffer to disk."""
        # Convert to atoms
        atoms_iter = (metadata_to_atoms(s) for s in items)
        # Use append mode
        self.dataset_manager.save_iter(
            atoms_iter,
            self.validation_path,
            mode="ab",
            calculate_checksum=False # Checksum for append is expensive/complex, skipping for speed
        )


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

        engine_cls: type[DynamicsEngine] = LAMMPSEngine
        if config.dynamics_engine.engine == "eon":
            engine_cls = EONEngine

        self.dynamics_engine: DynamicsEngine = dynamics_engine or _create_default_module(
            engine_cls, config
        )

        # If running in full mock mode (e.g. Oracle is mock), use MockValidator?
        # Or let Validator handle it. Validator now calls check_phonons which needs a real calculator.
        # If we are in mock mode, we don't have a real potential file usually.
        # So we should use MockValidator if oracle.mock is True?

        val_cls: type[ValidatorInterface] = Validator
        if config.oracle.mock:
             from pyacemaker.modules.validator import MockValidator
             val_cls = MockValidator

        self.validator: ValidatorInterface = validator or _create_default_module(val_cls, config)

        # State
        self.current_potential: Potential | None = None
        # Dataset is now file-based to prevent OOM
        # Use config option instead of hardcoded
        self.dataset_path = (
            self.config.project.root_dir / "data" / self.config.orchestrator.dataset_file
        )
        # Validation set path
        self.validation_path = (
            self.config.project.root_dir / "data" / CONSTANTS.default_validation_file
        )
        # Training set path (persistent subset)
        self.training_path = (
            self.config.project.root_dir / "data" / CONSTANTS.default_training_file
        )

        self.dataset_manager = DatasetManager()
        self.cycle_count = 0
        self.processed_items_count = 0

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
                    status=CycleStatus.FAILED,
                    metrics=Metrics.model_validate({"cycles": self.cycle_count}),
                )

        self.logger.info("Pipeline completed")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"cycles": self.cycle_count}),
        )

    def _save_dataset_stream(self, stream: Iterator[StructureMetadata]) -> None:
        """Convert metadata stream to atoms and save to dataset.

        Optimized to skip expensive checksum calculation during active learning loop.
        Removes stale checksum file to prevent validation failures.
        """
        atoms_stream = (metadata_to_atoms(s) for s in stream)
        # Skip checksum calculation for O(1) append
        self.dataset_manager.save_iter(
            atoms_stream,
            self.dataset_path,
            mode="ab",
            calculate_checksum=False
        )
        # Remove stale checksum file if it exists, as it no longer matches the dataset
        checksum_path = self.dataset_path.with_suffix(self.dataset_path.suffix + ".sha256")
        if checksum_path.exists():
            try:
                checksum_path.unlink()
            except OSError:
                self.logger.warning("Failed to remove stale checksum file.")

    def _run_cold_start(self) -> None:
        """Execute cold start to generate initial dataset."""
        initial_structures = self.structure_generator.generate_initial_structures()
        # Stream: initial_structures (iter) -> compute_batch (iter) -> save (append)
        computed_stream = self.oracle.compute_batch(initial_structures)
        self._save_dataset_stream(computed_stream)

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
        """Execute training phase with incremental partitioning."""
        self.logger.info(f"Phase: Training (Incremental from index {self.processed_items_count})")

        # 1. Incremental Partitioning
        # Only process new items from dataset
        splitter = DatasetSplitter(
            self.dataset_path,
            self.validation_path,
            self.dataset_manager,
            self.config.orchestrator.validation_split,
            self.config.orchestrator.max_validation_size,
            buffer_size=self.config.orchestrator.validation_buffer_size,
            start_index=self.processed_items_count
        )

        # Iterate splitter to identify new training items
        new_training_items = splitter.train_stream()

        # Append new training items to persistent training set
        atoms_iter = (metadata_to_atoms(s) for s in new_training_items)
        self.dataset_manager.save_iter(
            atoms_iter,
            self.training_path,
            mode="ab",
            calculate_checksum=False # Streaming append
        )

        # Update progress tracking
        added_count = splitter.processed_count
        self.processed_items_count += added_count
        self.logger.info(f"Processed {added_count} new items. Total processed: {self.processed_items_count}")

        # 2. Train on Full Training Set
        if not self.training_path.exists():
            self.logger.warning("No training data available.")
            return

        # Pass iterator over persistent training set to trainer
        # This re-reads the full training set (O(N_train)) which is necessary for retraining,
        # but avoids O(N_total) splitting logic.

        # Need to convert atoms back to metadata for Trainer interface
        def training_stream() -> Iterator[StructureMetadata]:
            # Use buffer_size for efficiency
            for atoms in self.dataset_manager.load_iter(self.training_path):
                yield atoms_to_metadata(atoms)

        potential = self.trainer.train(training_stream(), self.current_potential)
        self.current_potential = potential

    def _run_validation_phase(self) -> bool:
        """Execute validation phase.

        Returns:
            bool: True if validation passed, False otherwise.
        """
        self.logger.info("Phase: Validation")
        if not self.current_potential:
            self.logger.warning("No potential to validate.")
            return False

        if not self.validation_path.exists():
            self.logger.warning("Empty validation set.")
            return True

        # Load validation set from file
        # Streaming to Validator to avoid OOM
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
        self._save_dataset_stream(new_data)
