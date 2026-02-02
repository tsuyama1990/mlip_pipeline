import logging
import shutil
from pathlib import Path

from mlip_autopipec.config import Config
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.state import StateManager

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: Config,
        explorer: Explorer,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator,
    ) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator

        self.state_manager = StateManager(Path("state.json"))

        # Initialize or load state
        loaded_state = self.state_manager.load()
        if loaded_state:
            self.state = loaded_state
            logger.info(f"Resumed from iteration {self.state.iteration}")
        else:
            self.state = WorkflowState()
            logger.info("Initialized new workflow state")

    def _merge_datasets(self, cycle_dir: Path, new_data_paths: list[Path]) -> Path:
        """
        Merges new data into the training dataset.

        Since we don't have ASE or other tools installed, this is a mock implementation
        that simply 'touches' a new dataset file or copies the old one.
        """
        logger.info(f"Merging {len(new_data_paths)} new data files into dataset.")

        # Original dataset path
        current_dataset = self.config.training.dataset_path

        # New dataset path
        updated_dataset = cycle_dir / "dataset_updated.pckl"

        # In a real implementation, we would load current_dataset, load new_data_paths,
        # concatenate them, and save to updated_dataset.
        # For now, we just copy the original or touch a new one.
        if current_dataset.exists():
            shutil.copy(current_dataset, updated_dataset)
        else:
            updated_dataset.touch()

        return updated_dataset

    def run(self) -> None:
        """Executes the Active Learning Cycle."""
        while self.state.iteration < self.config.orchestrator.max_iterations:
            current_iter = self.state.iteration
            logger.info(f"Starting Cycle {current_iter}")

            cycle_dir = Path(f"active_learning/iter_{current_iter:03d}")
            cycle_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. Exploration
                current_pot_obj = None
                if self.state.current_potential_path:
                     current_pot_obj = Potential(
                         path=self.state.current_potential_path,
                         generation_id=current_iter - 1 if current_iter > 0 else 0
                     )

                exploration_result = self.explorer.explore(
                    potential=current_pot_obj,
                    work_dir=cycle_dir / "exploration"
                )

                # Check if we found anything new
                candidates = exploration_result.get("candidates", [])
                if not candidates:
                    logger.info("Exploration found no new structures. Converged?")
                    break

                # 2. Oracle (Selection & Calculation)
                new_data_paths = self.oracle.compute(
                    structures=candidates,
                    work_dir=cycle_dir / "oracle"
                )

                if not new_data_paths:
                    logger.warning("Oracle produced no data.")

                # 3. Training
                logger.info("Starting Training Phase")

                # Merge new data
                updated_dataset_path = self._merge_datasets(cycle_dir, new_data_paths)

                potential_path = self.trainer.train(
                    dataset=updated_dataset_path,
                    previous_potential=self.state.current_potential_path,
                    output_dir=cycle_dir / "training"
                )

                # Rename/Copy to preserve history
                unique_filename = f"potential_iter_{current_iter}.yace"
                unique_path = Path(unique_filename)

                if potential_path.exists():
                    shutil.copy(potential_path, unique_path)

                # 4. Validation
                val_result = None
                if self.config.validation.run_validation:
                    val_result = self.validator.validate(unique_path)
                    if not val_result.get("passed", False):
                        logger.warning(f"Validation failed: {val_result}")

                # Update state
                self.state.current_potential_path = unique_path
                self.state.history.append(
                    {
                        "iteration": current_iter,
                        "potential": str(unique_path),
                        "status": "success",
                        "validation": val_result
                    }
                )

                # Increment iteration
                self.state.iteration += 1
                self.state_manager.save(self.state)
                logger.info(f"Cycle {current_iter} completed")

            except Exception:
                logger.exception(f"Cycle {current_iter} failed")
                raise
