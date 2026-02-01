import logging
from pathlib import Path
from typing import Optional

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult, JobStatus
from mlip_autopipec.orchestration.phases import (
    PhaseExploration,
    PhaseDetection,
    PhaseSelection,
    PhaseRefinement,
    PhaseValidation
)

logger = logging.getLogger("mlip_autopipec")


class Orchestrator:
    """
    Orchestrates the active learning pipeline using specialized Phase strategies.
    Implements the Active Learning Cycle: Explore -> Detect -> Select -> Refine -> Validate.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.iteration = 0

        # Setup infrastructure
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Track current potential
        self.current_potential_path: Optional[Path] = None
        if self.config.training and self.config.training.initial_potential:
             self.current_potential_path = self.config.training.initial_potential

        # Initialize Phases
        self.phase_exploration = PhaseExploration(config)
        self.phase_detection = PhaseDetection(config)
        self.phase_selection = PhaseSelection(config)
        self.phase_refinement = PhaseRefinement(config, self.data_dir)
        self.phase_validation = PhaseValidation(config)

    def run_pipeline(self) -> JobResult:
        """
        Execute the pipeline configured in Config.
        """
        logger.info("Starting Active Learning Pipeline")

        max_iterations = self.config.orchestrator.max_iterations
        last_result: Optional[JobResult] = None

        while self.iteration < max_iterations:
            self.iteration += 1
            iter_dir = Path(f"active_learning/iter_{self.iteration:03d}")
            iter_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Iteration {self.iteration}/{max_iterations} ---")

            # 1. Explore (Structure Gen + MD)
            explore_result = self.phase_exploration.execute(iter_dir, self.current_potential_path)
            last_result = explore_result

            # 2. Detect
            if self.phase_detection.execute(explore_result):
                logger.info("High uncertainty detected. Proceeding to Refinement.")

                # 3. Select
                candidates = self.phase_selection.execute(explore_result)
                logger.info(f"Selected {len(candidates)} candidates for DFT.")

                if not candidates:
                    logger.warning("No candidates selected despite detection. Convergence?")
                    break

                # 4. Refine
                try:
                    new_potential = self.phase_refinement.execute(
                        candidates,
                        iter_dir,
                        self.current_potential_path
                    )
                    self.current_potential_path = new_potential
                    logger.info(f"Refinement complete. New potential: {new_potential}")
                except Exception as e:
                    logger.error(f"Refinement failed: {e}")
                    # If refinement fails, we break the loop to avoid infinite error loops
                    last_result = JobResult(
                        job_id="refine_fail",
                        status=JobStatus.FAILED,
                        work_dir=iter_dir,
                        duration_seconds=0.0,
                        log_content=str(e)
                    )
                    break

                # 5. Validate
                if self.iteration % self.config.orchestrator.validation_frequency == 0:
                    try:
                        self.phase_validation.execute(self.current_potential_path)
                    except Exception as e:
                        logger.error(f"Validation failed: {e}")

            else:
                logger.info("No high uncertainty detected. Convergence achieved.")
                break

        if last_result:
            logger.info(f"Pipeline Finished with status: {last_result.status.value}")
            return last_result

        return JobResult(
            job_id="noop",
            status=JobStatus.COMPLETED,
            work_dir=self.config.logging.file_path.parent,
            duration_seconds=0.0,
            log_content="No operations performed.",
        )
