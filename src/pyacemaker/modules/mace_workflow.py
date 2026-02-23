"""MACE Distillation Workflow Module."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from loguru import logger

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.dataset import SeedSelector
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
)
from pyacemaker.core.utils import (
    atoms_to_metadata,
    metadata_to_atoms,
    stream_metadata_to_atoms,
)
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    StructureStatus,
)
from pyacemaker.generator.direct import DirectGenerator
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.modules.oracle import MaceSurrogateOracle
from pyacemaker.oracle.dataset import DatasetManager


class MaceDistillationWorkflow:
    """Implements the 7-Step MACE Knowledge Distillation Workflow."""

    def __init__(
        self,
        config: PYACEMAKERConfig,
        dataset_manager: DatasetManager,
        dataset_path: Path,
        oracle: Oracle,
        trainer: Trainer,
        mace_trainer: Trainer,
        dynamics_engine: DynamicsEngine,
        structure_generator: StructureGenerator,
        validation_path: Path,  # For seed selection
        training_path: Path,    # For seed selection
    ) -> None:
        """Initialize the workflow."""
        self.config = config
        self.logger = logger.bind(name="MaceWorkflow")
        self.dataset_manager = dataset_manager
        self.dataset_path = dataset_path
        self.oracle = oracle
        self.trainer = trainer
        self.mace_trainer = mace_trainer
        self.dynamics_engine = dynamics_engine
        self.structure_generator = structure_generator
        self.validation_path = validation_path
        self.training_path = training_path

    def run(self) -> ModuleResult:
        """Run the workflow."""
        if not isinstance(self.oracle, UncertaintyModel):
            msg = "Oracle must implement UncertaintyModel for MACE distillation."
            raise TypeError(msg)

        dist_config = self.config.distillation

        # Step 1: DIRECT Sampling
        pool_path = self._step1_direct_sampling(dist_config)

        # Step 2 & 3: Active Learning & Fine-tuning
        fine_tuned_potential = self._step2_active_learning_loop(dist_config, pool_path)

        if not fine_tuned_potential:
            # Fallback to configured model if no fine-tuning happened
            self.logger.warning("No fine-tuning performed. Using base model from config.")
            fine_tuned_potential = Potential(
                path=Path(self.config.oracle.mace.model_path if self.config.oracle.mace else "mock"),
                type=PotentialType.MACE,
                version="1.0",
                metrics={},
                parameters={},
            )

        # Step 4: Surrogate Data Generation
        surrogate_structures_path = self._step4_surrogate_data_generation(
            dist_config, fine_tuned_potential
        )

        # Step 5: Surrogate Labeling
        # Update oracle model first!
        if hasattr(self.oracle, "update_model"):
            self.oracle.update_model(fine_tuned_potential.path)  # type: ignore[attr-defined]

        surrogate_dataset_path = self._step5_surrogate_labeling(surrogate_structures_path)

        # Step 6: Pacemaker Base Training
        base_ace_potential = self._step6_pacemaker_base_training(surrogate_dataset_path)

        # Step 7: Delta Learning
        final_potential = self._step7_delta_learning(dist_config, base_ace_potential)

        return ModuleResult(
            status="success",
            metrics=Metrics(),
            artifacts={"potential": str(final_potential.path)}
        )

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

    def _step2_active_learning_loop(
        self, dist_config: Any, pool_path: Path
    ) -> Potential | None:
        """Step 2 & 3: MACE Uncertainty-based Active Learning & Fine-tuning."""
        self.logger.info("Step 2: MACE Active Learning Loop")

        calculated_ids: set[Any] = set()
        current_potential: Potential | None = None

        # Configured iterations
        max_cycles = dist_config.step2_active_learning.cycles
        for i in range(max_cycles):
            self.logger.info(f"Step 2 (Iteration {i + 1}/{max_cycles})")

            # Update oracle if we have a new potential
            if current_potential and hasattr(self.oracle, "update_model"):
                self.oracle.update_model(current_potential.path)  # type: ignore[attr-defined]

            potential = self._execute_active_learning_iteration(
                dist_config, pool_path, calculated_ids, current_potential
            )

            if potential is None:
                self.logger.info("Iteration stopped (no candidates or convergence).")
                break

            current_potential = potential

        return current_potential

    def _execute_active_learning_iteration(
        self,
        dist_config: Any,
        pool_path: Path,
        calculated_ids: set[Any],
        current_potential: Potential | None
    ) -> Potential | None:
        """Execute a single iteration of Active Learning."""
        # Load pool
        pool_iter = (
            atoms_to_metadata(a) for a in self.dataset_manager.load_iter(pool_path)
        )
        # Filter out calculated IDs
        unknown_pool = (
            s
            for s in pool_iter
            if s.status != StructureStatus.CALCULATED and s.id not in calculated_ids
        )

        # Compute uncertainty
        uncertainty_oracle: UncertaintyModel = self.oracle  # type: ignore[assignment]
        scored_pool = uncertainty_oracle.compute_uncertainty(unknown_pool)

        # Select Top N
        n_select = dist_config.step2_active_learning.n_select
        threshold = dist_config.step2_active_learning.uncertainty_threshold

        learner = ActiveLearner()
        selected = learner.select_batch(scored_pool, n_select, threshold=threshold)

        if not selected:
            self.logger.info("No candidates selected (threshold not met or pool empty).")
            return None

        # Mark selected as calculated
        for s in selected:
            calculated_ids.add(s.id)

        # Compute DFT
        self.logger.info(f"Computing DFT for {len(selected)} structures")
        computed_iter = self.oracle.compute_batch(selected)
        self._save_dataset_stream(computed_iter)

        # Fine-tune MACE
        self.logger.info("Fine-tuning MACE...")

        def train_stream() -> Iterator[StructureMetadata]:
            yield from (
                atoms_to_metadata(a)
                for a in self.dataset_manager.load_iter(self.dataset_path)
            )

        return self.mace_trainer.train(train_stream(), initial_potential=current_potential)

    def _step4_surrogate_data_generation(
        self, dist_config: Any, fine_tuned_potential: Potential
    ) -> Path:
        """Step 4: Surrogate Data Generation."""
        self.logger.info("Step 4: Surrogate Data Generation")

        # Use fine-tuned potential
        # Seeds from dataset
        seeds = self._get_exploration_seeds(n_seeds=5)
        surrogate_iter = self.dynamics_engine.run_exploration(fine_tuned_potential, seeds)

        # Stream structures directly to file
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

        # We assume self.oracle is already updated with fine-tuned potential in run() method
        # But if it's not MaceSurrogateOracle (e.g. MockOracle), this might behave differently.
        # But for MACE workflow, we expect MaceSurrogateOracle logic (or Mock mimicking it).

        mace_labeler = self.oracle

        # Note: Previous code instantiated a NEW MaceSurrogateOracle:
        # mace_labeler = MaceSurrogateOracle(self.config)
        # But we want to use the one we injected (and updated).
        # However, MaceSurrogateOracle inherits Oracle interface.

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
        """Convert metadata stream to atoms and save to dataset."""
        atoms_stream = stream_metadata_to_atoms(stream)
        self.dataset_manager.save_iter(
            atoms_stream, self.dataset_path, mode="ab", calculate_checksum=False
        )
        # Remove stale checksum file if it exists
        checksum_path = self.dataset_path.with_suffix(self.dataset_path.suffix + ".sha256")
        if checksum_path.exists():
            try:
                checksum_path.unlink()
            except OSError:
                self.logger.warning("Failed to remove stale checksum file.")

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
