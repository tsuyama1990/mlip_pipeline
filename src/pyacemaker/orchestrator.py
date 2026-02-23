"""Orchestrator module implementation."""

from collections.abc import Callable, Iterable, Iterator
from itertools import chain, islice
from pathlib import Path
from typing import Any, TypeVar

from pyacemaker.core.base import BaseModule, Metrics, ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.dataset import DatasetSplitter, SeedSelector
from pyacemaker.core.interfaces import (
    CycleResult,
    DynamicsEngine,
    IOrchestrator,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
)
from pyacemaker.core.interfaces import (
    Validator as ValidatorInterface,
)
from pyacemaker.core.utils import (
    atoms_to_metadata,
    metadata_to_atoms,
)
from pyacemaker.domain_models.models import (
    CycleStatus,
    Potential,
    PotentialType,
    StructureMetadata,
    StructureStatus,
)
from pyacemaker.generator.direct import DirectGenerator
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.modules.dynamics_engine import EONEngine, LAMMPSEngine
from pyacemaker.modules.oracle import DFTOracle, MaceSurrogateOracle, MockOracle
from pyacemaker.modules.structure_generator import (
    AdaptiveStructureGenerator,
    RandomStructureGenerator,
)
from pyacemaker.modules.trainer import MaceTrainer, PacemakerTrainer
from pyacemaker.modules.validator import Validator
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
        mace_trainer: Trainer | None = None,
    ) -> None:
        """Initialize the orchestrator and sub-modules."""
        super().__init__(config)
        self.config = config

        # Dependency Injection with fallbacks using factory pattern
        # Logic for selecting structure generator strategy
        if structure_generator:
            self.structure_generator = structure_generator
        else:
            sg_cls = (
                AdaptiveStructureGenerator
                if config.structure_generator.strategy == "adaptive"
                else RandomStructureGenerator
            )
            self.structure_generator = _create_default_module(sg_cls, config)

        # Select Oracle implementation based on config
        oracle_cls: type[Oracle]
        if config.oracle.mace:
            oracle_cls = MaceSurrogateOracle
        elif config.oracle.mock:
            oracle_cls = MockOracle
        else:
            oracle_cls = DFTOracle

        self.oracle: Oracle = oracle or _create_default_module(oracle_cls, config)

        self.trainer: Trainer = trainer or _create_default_module(PacemakerTrainer, config)

        # Initialize MaceTrainer if needed for distillation
        self.mace_trainer: Trainer | None = mace_trainer
        if not self.mace_trainer and config.distillation.enable_mace_distillation:
            self.mace_trainer = _create_default_module(MaceTrainer, config)

        engine_cls: type[DynamicsEngine] = LAMMPSEngine
        if config.dynamics_engine.engine == "eon":
            engine_cls = EONEngine

        self.dynamics_engine: DynamicsEngine = dynamics_engine or _create_default_module(
            engine_cls, config
        )

        val_cls: type[ValidatorInterface] = Validator
        if config.oracle.mock:
            from pyacemaker.modules.validator import MockValidator

            val_cls = MockValidator

        self.validator: ValidatorInterface = validator or _create_default_module(val_cls, config)

        # State
        self.current_potential: Potential | None = None
        # Dataset is now file-based to prevent OOM
        self.dataset_path = (
            self.config.project.root_dir / "data" / self.config.orchestrator.dataset_file
        )
        # Validation set path
        self.validation_path = (
            self.config.project.root_dir / "data" / self.config.orchestrator.validation_file
        )
        # Training set path (persistent subset)
        self.training_path = self.config.project.root_dir / "data" / CONSTANTS.default_training_file

        self.dataset_manager = DatasetManager()
        self.cycle_count = 0
        self.processed_items_count = 0

    def run(self) -> ModuleResult:
        """Run the active learning pipeline."""
        self.logger.info("Starting Active Learning Pipeline")

        # Check for MACE Distillation Mode
        if self.config.distillation.enable_mace_distillation:
            self.logger.info("Mode: MACE Distillation Workflow")
            return self._run_mace_distillation()

        # Default: Classic Active Learning
        self.logger.info("Mode: Standard Active Learning Loop")
        return self._run_active_learning_loop()

    def run_cycle(self) -> CycleResult:
        """Execute one active learning cycle (Standard Loop)."""
        # This method is kept for interface compliance and used by _run_active_learning_loop
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

    def _run_active_learning_loop(self) -> ModuleResult:
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

    def _run_mace_distillation(self) -> ModuleResult:
        """Run the 7-Step MACE Distillation Workflow."""
        if not isinstance(self.oracle, UncertaintyModel):
            msg = "Oracle must implement UncertaintyModel for MACE distillation."
            raise TypeError(msg)

        dist_config = self.config.distillation

        # Step 1: DIRECT Sampling
        pool_path = self._step1_direct_sampling(dist_config)

        # Step 2 & 3: Active Learning & Fine-tuning
        self._step2_active_learning_loop(dist_config, pool_path)

        # Step 4: Surrogate Data Generation
        surrogate_structures_path = self._step4_surrogate_data_generation(dist_config)

        # Step 5: Surrogate Labeling
        surrogate_dataset_path = self._step5_surrogate_labeling(surrogate_structures_path)

        # Step 6: Pacemaker Base Training
        base_ace_potential = self._step6_pacemaker_base_training(surrogate_dataset_path)

        # Step 7: Delta Learning
        self.current_potential = self._step7_delta_learning(
            dist_config, base_ace_potential
        )

        return ModuleResult(status="success", metrics=Metrics(), artifacts={})

    def _step1_direct_sampling(self, dist_config: Any) -> Path:
        """Step 1: DIRECT Sampling (Entropy Maximization)."""
        self.logger.info("Step 1: DIRECT Sampling")

        # Use DirectGenerator specifically
        direct_generator = DirectGenerator(self.config)

        samples_iter = direct_generator.generate_direct_samples(
            n_samples=dist_config.step1_direct_sampling.target_points,
            objective=dist_config.step1_direct_sampling.objective,
        )
        pool_path = (
            self.config.project.root_dir / "data" / "pool_structures.pckl.gzip"
        )
        self.dataset_manager.save_iter(
            (metadata_to_atoms(s) for s in samples_iter),
            pool_path,
            calculate_checksum=False,
        )
        self.logger.info(f"Generated pool at {pool_path}")
        return pool_path

    def _step2_active_learning_loop(self, dist_config: Any, pool_path: Path) -> None:
        """Step 2 & 3: MACE Uncertainty-based Active Learning & Fine-tuning."""
        self.logger.info("Step 2: MACE Active Learning Loop")

        if not self.mace_trainer:
            msg = "MaceTrainer not initialized."
            raise RuntimeError(msg)

        calculated_ids: set[Any] = set()

        # Configured iterations
        max_cycles = dist_config.step2_active_learning.cycles
        for i in range(max_cycles):
            self.logger.info(f"Step 2 (Iteration {i + 1}/{max_cycles})")

            # Load pool
            pool_iter = (
                atoms_to_metadata(a)
                for a in self.dataset_manager.load_iter(pool_path)
            )
            # Filter out calculated IDs
            # (Note: pool file itself is not updated, so we filter in memory using set)
            unknown_pool = (
                s
                for s in pool_iter
                if s.status != StructureStatus.CALCULATED and s.id not in calculated_ids
            )

            # Compute uncertainty
            # Cast oracle to UncertaintyModel for mypy (validated at start of method)
            uncertainty_oracle: UncertaintyModel = self.oracle  # type: ignore[assignment]
            scored_pool = uncertainty_oracle.compute_uncertainty(unknown_pool)

            # Select Top N (using ActiveLearner)
            n_select = dist_config.step2_active_learning.n_select
            threshold = dist_config.step2_active_learning.uncertainty_threshold

            learner = ActiveLearner()
            selected = learner.select_batch(
                scored_pool,
                n_select,
                threshold=threshold
            )

            if not selected:
                self.logger.info("No candidates selected (threshold not met or pool empty).")
                break

            # Mark selected as calculated in local tracker
            for s in selected:
                calculated_ids.add(s.id)

            # Compute DFT
            self.logger.info(f"Computing DFT for {len(selected)} structures")
            computed_iter = self.oracle.compute_batch(selected)
            self._save_dataset_stream(computed_iter)

            # Fine-tune MACE (Step 3 integrated)
            self.logger.info("Fine-tuning MACE...")

            def train_stream() -> Iterator[StructureMetadata]:
                yield from (
                    atoms_to_metadata(a)
                    for a in self.dataset_manager.load_iter(self.dataset_path)
                )

            _ = self.mace_trainer.train(train_stream())

    def _step4_surrogate_data_generation(
        self, dist_config: Any
    ) -> Path:
        """Step 4: Surrogate Data Generation.

        Returns:
            Path to the generated surrogate dataset.
        """
        self.logger.info("Step 4: Surrogate Data Generation")
        # Reuse DynamicsEngine
        # We need a MACE potential object.
        # Construct generic potential for MACE
        mace_pot = Potential(
            path=Path(self.config.oracle.mace.model_path),  # type: ignore[union-attr]
            type=PotentialType.MACE,
            version="1.0",
            metrics={},
            parameters={},
        )

        # Seeds from dataset
        seeds = self._get_exploration_seeds(n_seeds=5)
        surrogate_iter = self.dynamics_engine.run_exploration(mace_pot, seeds)

        # Stream 1000 structures directly to file without list materialization
        surrogate_dataset_path = (
            self.config.project.root_dir / "data" / "surrogate_unlabeled.pckl.gzip"
        )

        limited_iter = islice(
            surrogate_iter, dist_config.step4_surrogate_sampling.target_points
        )

        self.dataset_manager.save_iter(
            (metadata_to_atoms(s) for s in limited_iter),
            surrogate_dataset_path,
            calculate_checksum=False,
        )
        self.logger.info(f"Generated surrogate dataset at {surrogate_dataset_path}")
        return surrogate_dataset_path

    def _step5_surrogate_labeling(
        self, surrogate_path: Path
    ) -> Path:
        """Step 5: Surrogate Labeling."""
        self.logger.info("Step 5: Surrogate Labeling")
        mace_labeler = MaceSurrogateOracle(self.config)  # Config has oracle.mace

        # Load stream
        def load_stream() -> Iterator[StructureMetadata]:
            for atoms in self.dataset_manager.load_iter(surrogate_path):
                yield atoms_to_metadata(atoms)

        labeled_surrogate_iter = mace_labeler.compute_batch(load_stream())

        # Save to separate "surrogate_dataset"
        surrogate_dataset_path = (
            self.config.project.root_dir / "data" / "surrogate_dataset.pckl.gzip"
        )
        self.dataset_manager.save_iter(
            (metadata_to_atoms(s) for s in labeled_surrogate_iter),
            surrogate_dataset_path,
            calculate_checksum=False,
        )
        return surrogate_dataset_path

    def _step6_pacemaker_base_training(
        self, surrogate_dataset_path: Path
    ) -> Potential:
        """Step 6: Pacemaker Base Training."""
        self.logger.info("Step 6: Pacemaker Base Training")

        def surrogate_train_stream() -> Iterator[StructureMetadata]:
            yield from (
                atoms_to_metadata(a)
                for a in self.dataset_manager.load_iter(surrogate_dataset_path)
            )

        return self.trainer.train(surrogate_train_stream())

    def _step7_delta_learning(
        self, dist_config: Any, base_potential: Potential
    ) -> Potential:
        """Step 7: Delta Learning (Fine-tuning with DFT)."""
        self.logger.info("Step 7: Delta Learning (Fine-tuning with DFT)")
        if dist_config.step7_pacemaker_finetune.enable:
            def dft_train_stream() -> Iterator[StructureMetadata]:
                yield from (
                    atoms_to_metadata(a)
                    for a in self.dataset_manager.load_iter(self.dataset_path)
                )

            weight_dft = dist_config.step7_pacemaker_finetune.weight_dft
            self.logger.info(f"Using DFT weight: {weight_dft}")

            return self.trainer.train(
                dft_train_stream(),
                initial_potential=base_potential,
                weight_dft=weight_dft,
            )
        return base_potential

    def _save_dataset_stream(self, stream: Iterator[StructureMetadata]) -> None:
        """Convert metadata stream to atoms and save to dataset.

        Optimized to skip expensive checksum calculation during active learning loop.
        Removes stale checksum file to prevent validation failures.
        """
        atoms_stream = (metadata_to_atoms(s) for s in stream)
        # Skip checksum calculation for O(1) append
        self.dataset_manager.save_iter(
            atoms_stream, self.dataset_path, mode="ab", calculate_checksum=False
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
        # Only process new items from dataset
        splitter = DatasetSplitter(
            self.dataset_path,
            self.validation_path,
            self.dataset_manager,
            self.config.orchestrator.validation_split,
            self.config.orchestrator.max_validation_size,
            buffer_size=self.config.orchestrator.validation_buffer_size,
            start_index=self.processed_items_count,
        )

        # Iterate splitter to identify new training items
        new_training_items = splitter.train_stream()

        # Append new training items to persistent training set
        atoms_iter = (metadata_to_atoms(s) for s in new_training_items)
        self.dataset_manager.save_iter(
            atoms_iter,
            self.training_path,
            mode="ab",
            calculate_checksum=False,  # Streaming append
        )

        # Update progress tracking
        added_count = splitter.processed_count
        self.processed_items_count += added_count
        self.logger.info(
            f"Processed {added_count} new items. Total processed: {self.processed_items_count}"
        )

        # 2. Train on Full Training Set
        if not self.training_path.exists():
            self.logger.warning("No training data available.")
            return

        # Pass iterator over persistent training set to trainer
        # This re-reads the full training set (O(N_train)) which is necessary for retraining,
        # but avoids O(N_total) splitting logic.

        # Need to convert atoms back to metadata for Trainer interface
        # load_iter is a generator, so this is already streaming.
        # We wrap it to convert Atoms -> StructureMetadata on the fly.
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

    def _get_exploration_seeds(self, n_seeds: int = 20) -> list[StructureMetadata]:
        """Get seed structures for exploration."""
        selector = SeedSelector(self.dataset_manager)
        seeds = selector.get_seeds(
            self.validation_path,
            self.training_path,
            self.structure_generator,
            n_seeds,
        )

        if not seeds:
            self.logger.warning("No seeds found in datasets or generator.")

        return seeds

    def _run_exploration_and_selection_phase(
        self,
    ) -> Iterable[StructureMetadata] | None:
        """Execute exploration and selection phases.

        Returns:
            Iterable of selected structures for calculation, or None if converged.
        """
        self.logger.info("Phase: Exploration")
        if not self.current_potential:
            return None

        # Select seeds for exploration
        seeds = self._get_exploration_seeds()
        if not seeds:
            self.logger.warning("No seeds available for exploration.")
            return None

        self.logger.info(f"Exploration starting with {len(seeds)} seeds.")

        # Run Dynamics Engine (Exploration)
        high_uncertainty_stream, max_gamma_container = self._run_exploration_stream(seeds)

        if high_uncertainty_stream is None:
             self.logger.info("No high uncertainty structures found. Converged?")
             return None

        # Selection (Local Candidates & Active Set)
        self.logger.info("Phase: Selection")
        active_set_structures = self._run_selection_phase(high_uncertainty_stream)

        # Log stats after consumption (Note: gamma will be updated as stream is consumed)
        # Since stream is consumed by selection phase, the value should be final here.
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

            # Check if we found anything without consuming the whole stream
            try:
                first_structure = next(high_uncertainty_iter)
            except StopIteration:
                return None, [0.0]

            # Reconstruct iterator
            high_uncertainty_stream = chain([first_structure], high_uncertainty_iter)

            # Spy on the stream to calculate metrics (max gamma) without materializing list
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
                    # Stop yielding on error, effectively truncating stream
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

        # Stream candidates generation
        candidates_iter = self.structure_generator.generate_batch_candidates(
            high_uncertainty_stream,
            n_candidates_per_seed=n_local,
            cycle=self.cycle_count,
        )

        n_select = self.config.orchestrator.n_active_set_select
        active_set = self.trainer.select_active_set(candidates_iter, n_select=n_select)

        # Active set now returns path, structures might be None to prevent OOM
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
        # Stream processing: iterable -> compute_batch(iter) -> save (append)
        new_data = self.oracle.compute_batch(structures)
        self._save_dataset_stream(new_data)
