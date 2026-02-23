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
    save_metadata_stream,
)
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    StructureStatus,
)
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.oracle.dataset import DatasetManager


class MaceDistillationWorkflow:
    """Implements the 7-Step MACE Knowledge Distillation Workflow."""

    def __init__(
        self,
        config: PYACEMAKERConfig,
        dataset_manager: DatasetManager,
        dataset_path: Path,
        oracle: Oracle,
        mace_oracle: UncertaintyModel,
        trainer: Trainer,
        mace_trainer: Trainer,
        dynamics_engine: DynamicsEngine,
        structure_generator: StructureGenerator,
        validation_path: Path,
        training_path: Path,
        active_learner: ActiveLearner | None = None,
    ) -> None:
        """Initialize the workflow."""
        self.config = config
        self.logger = logger.bind(name="MaceWorkflow")
        self.dataset_manager = dataset_manager
        self.dataset_path = dataset_path
        self.oracle = oracle
        self.mace_oracle = mace_oracle
        self.trainer = trainer
        self.mace_trainer = mace_trainer
        self.dynamics_engine = dynamics_engine
        self.structure_generator = structure_generator
        self.validation_path = validation_path
        self.training_path = training_path
        self.active_learner = active_learner or ActiveLearner()

    def run(self) -> ModuleResult:
        """Run the workflow."""
        try:
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
            # Update mace_oracle model first!
            if hasattr(self.mace_oracle, "update_model"):
                self.mace_oracle.update_model(fine_tuned_potential.path)

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
        except Exception as e:
            self.logger.exception("MACE Distillation Workflow failed")
            return ModuleResult(
                status="failed",
                metrics=Metrics(),
                error=str(e),
            )

    def _step1_direct_sampling(self, dist_config: Any) -> Path:
        """Step 1: DIRECT Sampling (Entropy Maximization)."""
        self.logger.info("Step 1: DIRECT Sampling")

        samples_iter = self.structure_generator.generate_direct_samples(
            n_samples=dist_config.step1_direct_sampling.target_points,
            objective=dist_config.step1_direct_sampling.objective,
        )

        pool_file = dist_config.pool_file
        pool_path = self.config.project.root_dir / "data" / pool_file

        save_metadata_stream(
            self.dataset_manager,
            samples_iter,
            pool_path,
            mode="wb",  # Overwrite pool
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

        max_cycles = dist_config.step2_active_learning.cycles
        for i in range(max_cycles):
            self.logger.info(f"Step 2 (Iteration {i + 1}/{max_cycles})")

            # Update MACE Oracle with latest potential
            if current_potential and hasattr(self.mace_oracle, "update_model"):
                self.mace_oracle.update_model(current_potential.path)

            potential = self._execute_active_learning_iteration(
                dist_config, pool_path, calculated_ids, current_potential
            )

            if potential is None:
                self.logger.info("Iteration stopped (no candidates or convergence).")
                break

            current_potential = potential

        return current_potential

    def _select_candidates(
        self,
        dist_config: Any,
        pool_path: Path,
        calculated_ids: set[Any]
    ) -> list[StructureMetadata] | None:
        """Select candidates using uncertainty sampling."""
        pool_iter = (
            atoms_to_metadata(a) for a in self.dataset_manager.load_iter(pool_path)
        )
        unknown_pool = (
            s
            for s in pool_iter
            if s.status != StructureStatus.CALCULATED and s.id not in calculated_ids
        )

        # Use MACE Oracle for uncertainty
        scored_pool = self.mace_oracle.compute_uncertainty(unknown_pool)

        n_select = dist_config.step2_active_learning.n_select
        threshold = dist_config.step2_active_learning.uncertainty_threshold

        return self.active_learner.select_batch(scored_pool, n_select, threshold=threshold)

    def _execute_active_learning_iteration(
        self,
        dist_config: Any,
        pool_path: Path,
        calculated_ids: set[Any],
        current_potential: Potential | None
    ) -> Potential | None:
        """Execute a single iteration of Active Learning."""
        selected = self._select_candidates(dist_config, pool_path, calculated_ids)

        if not selected:
            self.logger.info("No candidates selected (threshold not met or pool empty).")
            return None

        # Mark selected as calculated
        for s in selected:
            calculated_ids.add(s.id)

        # Compute DFT (Ground Truth) using primary Oracle
        self.logger.info(f"Computing DFT for {len(selected)} structures")
        computed_iter = self.oracle.compute_batch(selected)

        save_metadata_stream(
            self.dataset_manager,
            computed_iter,
            self.dataset_path,
            mode="ab",  # Append to dataset
            calculate_checksum=False,
        )

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

        seeds = self._get_exploration_seeds(n_seeds=5)
        surrogate_iter = self.dynamics_engine.run_exploration(fine_tuned_potential, seeds)

        surrogate_file = dist_config.surrogate_file
        surrogate_dataset_path = self.config.project.root_dir / "data" / surrogate_file

        limited_iter = islice(
            surrogate_iter, dist_config.step4_surrogate_sampling.target_points
        )

        save_metadata_stream(
            self.dataset_manager,
            limited_iter,
            surrogate_dataset_path,
            mode="wb",  # Overwrite surrogate pool
            calculate_checksum=False,
        )
        self.logger.info(f"Generated surrogate dataset at {surrogate_dataset_path}")
        return surrogate_dataset_path

    def _step5_surrogate_labeling(
        self, surrogate_path: Path
    ) -> Path:
        """Step 5: Surrogate Labeling."""
        self.logger.info("Step 5: Surrogate Labeling")

        # Use MACE Oracle for labeling
        mace_labeler = self.mace_oracle
        # Ensure it has compute_batch (inherited from Oracle/BaseOracle)
        if not isinstance(mace_labeler, Oracle):
            msg = "MACE Oracle does not support labeling (compute_batch)."
            raise TypeError(msg)

        dist_config = self.config.distillation

        def load_stream() -> Iterator[StructureMetadata]:
            for atoms in self.dataset_manager.load_iter(surrogate_path):
                yield atoms_to_metadata(atoms)

        labeled_surrogate_iter = mace_labeler.compute_batch(load_stream())

        surrogate_dataset_file = dist_config.surrogate_dataset_file
        surrogate_dataset_path = self.config.project.root_dir / "data" / surrogate_dataset_file

        save_metadata_stream(
            self.dataset_manager,
            labeled_surrogate_iter,
            surrogate_dataset_path,
            mode="wb",  # Overwrite labeled dataset
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
