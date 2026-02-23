"""MACE Distillation Workflow."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from loguru import logger

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
)
from pyacemaker.core.utils import atoms_to_metadata, metadata_to_atoms
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    StructureStatus,
)
from pyacemaker.modules.active_learner import ActiveLearner
from pyacemaker.oracle.dataset import DatasetManager


class MaceDistillationWorkflow:
    """MACE Distillation Workflow."""

    def __init__(
        self,
        config: PYACEMAKERConfig,
        dataset_manager: DatasetManager,
        structure_generator: StructureGenerator,
        oracle: Oracle,
        mace_trainer: Trainer,
        active_learner: ActiveLearner,
        uncertainty_model: UncertaintyModel,
        dynamics_engine: DynamicsEngine,
        trainer: Trainer,
    ) -> None:
        """Initialize the workflow."""
        self.config = config
        self.logger = logger.bind(name="MaceDistillation")
        self.dataset_manager = dataset_manager

        self.structure_generator = structure_generator
        self.oracle = oracle
        self.mace_trainer = mace_trainer
        self.active_learner = active_learner
        self.uncertainty_model = uncertainty_model
        self.dynamics_engine = dynamics_engine
        self.trainer = trainer

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

    def run(self) -> ModuleResult:
        """Run the 7-Step MACE Distillation Workflow."""
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
        samples_iter = self.structure_generator.generate_direct_samples(
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

        calculated_ids: set[Any] = set()

        max_cycles = dist_config.step2_active_learning.cycles
        for i in range(max_cycles):
            self.logger.info(f"Step 2 (Iteration {i + 1}/{max_cycles})")

            # Load pool
            pool_iter = (
                atoms_to_metadata(a)
                for a in self.dataset_manager.load_iter(pool_path)
            )
            # Filter out calculated IDs
            unknown_pool = (
                s
                for s in pool_iter
                if s.status != StructureStatus.CALCULATED and s.id not in calculated_ids
            )

            # Compute uncertainty (Stream)
            scored_pool = self.uncertainty_model.compute_uncertainty(unknown_pool)

            # Select Batch using ActiveLearner (Consumes stream, returns small list)
            selected = list(self.active_learner.select_batch(scored_pool))

            if not selected:
                self.logger.info("No more candidates selected.")
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

        # We assume self.config.oracle.mace exists and is valid if we reached here
        mace_model_path = Path(self.config.oracle.mace.model_path) if self.config.oracle.mace else Path("mace.model")

        mace_pot = Potential(
            path=mace_model_path,
            type=PotentialType.MACE,
            version="1.0",
            metrics={},
            parameters={},
        )

        seeds = self._get_exploration_seeds(n_seeds=5)
        surrogate_iter = self.dynamics_engine.run_exploration(mace_pot, seeds)

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
        # We reuse the uncertainty model if it's MACE, or create a labeler
        # Actually MaceSurrogateOracle implements compute_batch as well
        mace_labeler: Oracle = self.uncertainty_model  # type: ignore[assignment]
        # But wait, self.uncertainty_model is defined as UncertaintyModel interface.
        # It happens to be MaceSurrogateOracle which is also an Oracle.
        # We should cast or rely on dependency injection being correct.
        if not isinstance(mace_labeler, Oracle):
             msg = "Uncertainty model must also be an Oracle for step 5"
             raise TypeError(msg)

        def load_stream() -> Iterator[StructureMetadata]:
            for atoms in self.dataset_manager.load_iter(surrogate_path):
                yield atoms_to_metadata(atoms)

        labeled_surrogate_iter = mace_labeler.compute_batch(load_stream())

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

            return self.trainer.train(
                dft_train_stream(), initial_potential=base_potential
            )
        return base_potential

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

    def _get_exploration_seeds(self, n_seeds: int = 20) -> list[StructureMetadata]:
        """Get seed structures for exploration."""
        # Simple implementation for distillation step 4
        from pyacemaker.core.dataset import SeedSelector
        selector = SeedSelector(self.dataset_manager)
        return selector.get_seeds(
            self.validation_path,
            self.training_path,
            self.structure_generator,
            n_seeds,
        )
