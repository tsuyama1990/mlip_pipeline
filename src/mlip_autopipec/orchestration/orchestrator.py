import logging
from typing import Optional

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult, JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory

logger = logging.getLogger("mlip_autopipec")


class Orchestrator:
    """
    Orchestrates the active learning pipeline.
    Implements the Active Learning Cycle: Explore -> Detect -> Select -> Refine.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.iteration = 0
        self.labelled_structures: list[Structure] = []

    def run_pipeline(self) -> JobResult:
        """
        Execute the pipeline configured in Config.
        """
        logger.info("Starting Active Learning Pipeline")

        # For Cycle 02/03, we might default to 1 iteration if not specified
        max_iterations = self.config.orchestrator.max_iterations

        last_result: Optional[JobResult] = None

        while self.iteration < max_iterations:
            self.iteration += 1
            logger.info(f"--- Iteration {self.iteration}/{max_iterations} ---")

            # 1. Explore (Structure Gen + MD)
            last_result = self.explore()

            # 2. Detect
            if not self.detect(last_result):
                logger.info(
                    "No high uncertainty detected or loop finish. Convergence achieved."
                )
                break

            # 3. Select (Placeholder)
            self.select()

            # 4. Refine (Training)
            self.refine()

        if last_result:
            logger.info(f"Pipeline Finished with status: {last_result.status.value}")
            return last_result

        # Fallback if no iterations ran
        return JobResult(
            job_id="noop",
            status=JobStatus.COMPLETED,
            work_dir=self.config.logging.file_path.parent,
            duration_seconds=0.0,
            log_content="No operations performed.",
        )

    def explore(self) -> JobResult:
        """
        Phase 1: Exploration
        Generates structures and runs dynamics to explore the PES.
        """
        logger.info("Phase: Exploration (Structure Gen + MD)")

        # 1. Structure Generation
        gen_config = self.config.structure_gen
        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)
        logger.info(f"Generated structure: {structure.get_chemical_formula()}")

        # 2. Dynamics (MD)
        # In a full loop, we might use the latest potential.
        # Here we use the configured one (implied or internal LJ/ZBL).
        md_config = self.config.md
        runner = LammpsRunner(self.config.lammps)

        result = runner.run(structure, md_config)
        return result

    def detect(self, result: JobResult) -> bool:
        """
        Phase 2: Detection
        Analyze the job result for high uncertainty or errors.
        Returns True if retraining is needed (uncertainty detected).
        """
        logger.info("Phase: Detection")

        # Check if result has max_gamma (LammpsResult)
        if hasattr(result, "max_gamma") and getattr(result, "max_gamma") is not None:
            max_gamma = getattr(result, "max_gamma")
            if max_gamma > self.config.orchestrator.uncertainty_threshold:
                logger.warning(f"High uncertainty detected: {max_gamma}")
                return True

        # If we have labelled structures, we might want to train regardless of detection in this simple loop?
        # Or if the user wants to force training (max_iterations > 0 and we are in loop)
        # For testing purposes, let's return True if we have structures to train?
        # But detection should depend on UNCERTAINTY.

        # If we are in the loop, we usually want to proceed to Select/Refine if we found something.
        # If no uncertainty, we break.

        # For UAT/Demo, we might want to force it.
        # But strictly speaking, if max_gamma is low, we stop.
        return False

    def select(self) -> None:
        """
        Phase 3: Selection
        Select structures for DFT calculation.
        """
        logger.info("Phase: Selection (Placeholder)")
        # In future: Extract frames, run MaxVol, etc.

    def refine(self) -> None:
        """
        Phase 4: Refinement
        Run DFT and retrain potential.
        """
        logger.info("Phase: Refinement")

        if not self.labelled_structures:
            logger.info("No labelled structures to train on.")
            return

        if not self.config.training:
            logger.info("No training configuration found. Skipping training.")
            return

        # 2. Dataset Generation
        from mlip_autopipec.physics.training.dataset import DatasetManager
        from mlip_autopipec.physics.training.pacemaker import PacemakerRunner

        # Use a training subdirectory
        train_dir = self.config.logging.file_path.parent / f"training_{self.iteration}"
        dataset_path = train_dir / "train.pckl.gzip"

        ds_manager = DatasetManager()
        try:
            ds_path = ds_manager.atoms_to_dataset(
                self.labelled_structures, dataset_path
            )
        except RuntimeError as e:
            logger.error(f"Dataset creation failed: {e}")
            return

        # 3. Training
        runner = PacemakerRunner(work_dir=train_dir)

        # Active Set Selection
        if self.config.training.active_set_optimization:
            try:
                ds_path = runner.select_active_set(
                    ds_path, self.config.training, self.config.potential
                )
            except RuntimeError as e:
                logger.error(f"Active set selection failed: {e}")
                # Fallback to full dataset?
                pass

        # Train
        result = runner.train(ds_path, self.config.training, self.config.potential)
        if result.status == JobStatus.COMPLETED:
            logger.info(
                f"Training finished: RMSE Energy={result.validation_metrics.get('rmse_energy', 'N/A')}"
            )
        else:
            logger.error("Training failed.")
