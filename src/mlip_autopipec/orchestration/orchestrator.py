import logging
import shutil
from pathlib import Path

from mlip_autopipec.config import Config
from mlip_autopipec.domain_models.production import ProductionManifest
from mlip_autopipec.domain_models.workflow import HistoryEntry, WorkflowState
from mlip_autopipec.infrastructure.production import ProductionDeployer
from mlip_autopipec.orchestration.interfaces import (
    Explorer,
    Oracle,
    Selector,
    Trainer,
    Validator,
)
from mlip_autopipec.orchestration.state import StateManager

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: Config,
        explorer: Explorer,
        selector: Selector,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator | None = None,
    ) -> None:
        self.config = config
        self.state_manager = StateManager(Path("state.json"))

        self.explorer = explorer
        self.selector = selector
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator

        # Initialize or load state
        loaded_state = self.state_manager.load()
        if loaded_state:
            self.state = loaded_state
            logger.info(f"Resumed from iteration {self.state.iteration}")
        else:
            self.state = WorkflowState()
            logger.info("Initialized new workflow state")

    def run(self) -> None:
        """Executes the Active Learning Cycle."""
        while self.state.iteration < self.config.orchestrator.max_iterations:
            logger.info(f"Starting Cycle {self.state.iteration}")

            # Setup work directory for this iteration
            work_dir = Path(f"active_learning/iter_{self.state.iteration:03d}")
            work_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Phase 1: Exploration
                logger.info("Phase: Exploration")
                exploration_dir = work_dir / "exploration"
                exploration_dir.mkdir(parents=True, exist_ok=True)
                candidates = self.explorer.explore(
                    potential_path=self.state.current_potential_path,
                    work_dir=exploration_dir,
                )
                logger.info(f"Exploration found {len(candidates)} candidates")

                # Phase 2: Selection
                logger.info("Phase: Selection")
                selection_dir = work_dir / "selection"
                selection_dir.mkdir(parents=True, exist_ok=True)
                selected_candidates = self.selector.select(
                    candidates=candidates,
                    potential_path=self.state.current_potential_path,
                    work_dir=selection_dir,
                )
                logger.info(
                    f"Selection retained {len(selected_candidates)} candidates out of {len(candidates)}"
                )

                # Phase 3: Oracle
                logger.info("Phase: Oracle")
                # Ensure oracle directory exists
                oracle_dir = work_dir / "oracle"
                oracle_dir.mkdir(parents=True, exist_ok=True)
                new_data_paths = self.oracle.compute(
                    candidates=selected_candidates, work_dir=oracle_dir
                )
                logger.info(f"Oracle returned {len(new_data_paths)} new data files")

                # Phase 4: Training
                logger.info("Phase: Training")
                # Update dataset
                current_dataset = self.trainer.update_dataset(new_data_paths)

                training_dir = work_dir / "training"
                training_dir.mkdir(parents=True, exist_ok=True)

                potential_path = self.trainer.train(
                    dataset=current_dataset,
                    previous_potential=self.state.current_potential_path,
                    output_dir=training_dir,
                )
                logger.info(f"Training completed. Potential at {potential_path}")

                # Phase 5: Validation
                if self.validator and self.config.validation.run_validation:
                    logger.info("Phase: Validation")
                    validation_dir = work_dir / "validation"
                    validation_dir.mkdir(parents=True, exist_ok=True)
                    val_result = self.validator.validate(potential_path, validation_dir)
                    logger.info(f"Validation passed: {val_result.passed}")
                    if not val_result.passed:
                        logger.warning(f"Validation failed: {val_result.reason}")
                        # In strict mode, we might stop here. For now, we log and proceed or store status.

                # Finalize Cycle
                # Rename/Move potential to potentials/ directory to preserve history as per spec
                potentials_dir = Path("potentials")
                potentials_dir.mkdir(exist_ok=True)
                final_potential_path = (
                    potentials_dir / f"generation_{self.state.iteration:03d}.yace"
                )
                shutil.copy(potential_path, final_potential_path)

                # Update state
                self.state.current_potential_path = final_potential_path

                history_entry = HistoryEntry(
                    iteration=self.state.iteration,
                    potential_path=str(final_potential_path),
                    status="success",
                    candidates_count=len(candidates),
                    new_data_count=len(new_data_paths),
                )
                self.state.history.append(history_entry)

                # Increment iteration
                self.state.iteration += 1
                self.state_manager.save(self.state)
                logger.info(f"Cycle {self.state.iteration - 1} completed")

            except Exception:
                logger.exception(f"Cycle {self.state.iteration} failed")
                raise

        self.finalize()

    def finalize(self) -> None:
        """Deploy the final potential."""
        logger.info("Deploying Production Release...")

        if not self.state.current_potential_path:
            logger.warning("No potential to deploy.")
            return

        deployer = ProductionDeployer(self.config)

        # Estimate dataset size from history
        total_new_data = sum(h.new_data_count for h in self.state.history)

        last_metrics = {}
        if self.state.history:
            # Look for last entry with metrics
            for h in reversed(self.state.history):
                if h.metrics:
                    last_metrics = h.metrics
                    break

        manifest = ProductionManifest(
            version="1.0.0",
            author=self.config.project.name if hasattr(self.config.project, "name") else "AutoPipec",
            training_set_size=total_new_data,
            validation_metrics=last_metrics,
        )

        # Look for validation report
        report_path = Path("active_learning") / f"iter_{self.state.iteration - 1:03d}" / "validation" / "report.html"
        if not report_path.exists():
            report_path = Path("report.html")

        deployer.deploy(
            potential_path=self.state.current_potential_path,
            manifest=manifest,
            report_path=report_path,
            output_dir=Path(),
        )
