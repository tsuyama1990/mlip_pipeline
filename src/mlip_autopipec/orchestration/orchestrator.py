import logging
from typing import Optional, List

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult, JobStatus
from mlip_autopipec.domain_models.dynamics import LammpsResult
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.dft.qe_runner import QERunner
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner

logger = logging.getLogger("mlip_autopipec")


class Orchestrator:
    """
    Orchestrates the active learning pipeline.
    Implements the Active Learning Cycle: Explore -> Detect -> Select -> Refine.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.iteration = 0

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
                logger.info("No high uncertainty detected or loop finish. Convergence achieved.")
                break

            # 3. Select
            candidates = self.select(last_result)
            if not candidates:
                 logger.info("No candidates selected for refinement.")
                 break

            # 4. Refine
            last_result = self.refine(candidates)

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

        # If we have DFT and Training config, and we are in a loop (max_iterations > 1),
        # but no uncertainty info is available (e.g. initial run with LJ),
        # we might want to proceed to Refine anyway to create the first potential?
        # For now, strict adherence to logic: only if uncertainty detected.
        # But if max_iterations > 1 and we are at iter 1, maybe we force?
        # I'll stick to returning False if no signal.
        # But for UAT integration testing, I might need to mock this method or ensure LammpsResult has max_gamma.
        return False

    def select(self, result: JobResult) -> List[Structure]:
        """
        Phase 3: Selection
        Select structures for DFT calculation.
        """
        logger.info("Phase: Selection")
        candidates = []

        if isinstance(result, LammpsResult):
             # Simple strategy: take the final structure
             # In future: pick high uncertainty frames
             logger.info("Selecting final structure from MD trajectory.")
             candidates.append(result.final_structure)

        return candidates

    def refine(self, candidates: List[Structure]) -> JobResult:
        """
        Phase 4: Refinement
        Run DFT and retrain potential.
        """
        logger.info("Phase: Refinement")

        if not self.config.dft:
             raise ValueError("DFT configuration missing for refinement phase.")
        if not self.config.training:
             raise ValueError("Training configuration missing for refinement phase.")

        # 1. Labeling (DFT)
        labelled_structures = []
        dft_runner = QERunner()

        for i, struct in enumerate(candidates):
             logger.info(f"Running DFT for candidate {i+1}/{len(candidates)}")
             dft_result = dft_runner.run(struct, self.config.dft)

             if dft_result.status == JobStatus.COMPLETED:
                  # Update structure properties
                  s = struct.model_copy()
                  s.properties["energy"] = dft_result.energy
                  s.properties["forces"] = dft_result.forces
                  if dft_result.stress is not None:
                       s.properties["stress"] = dft_result.stress
                  labelled_structures.append(s)
             else:
                  logger.warning(f"DFT failed for candidate {i+1}: {dft_result.log_content}")

        if not labelled_structures:
             raise RuntimeError("No structures successfully labelled.")

        # 2. Learning (Training)
        # Use a sub-directory for training to avoid clutter
        # We need a work dir. Config has logging path, we can use its parent.
        base_work_dir = self.config.logging.file_path.parent
        train_work_dir = base_work_dir / f"train_iter_{self.iteration}"

        dataset_mgr = DatasetManager(train_work_dir)
        dataset_path = train_work_dir / "train.pckl.gzip"
        dataset_mgr.convert(labelled_structures, dataset_path)

        pace_runner = PacemakerRunner(train_work_dir)

        # Active Set Selection (Optional)
        if self.config.training.active_set_optimization:
             dataset_path = pace_runner.select_active_set(dataset_path)

        # Train
        train_result = pace_runner.train(
             dataset_path,
             self.config.training,
             self.config.potential
        )

        logger.info(f"Training completed. New potential: {train_result.potential.path}")
        return train_result
