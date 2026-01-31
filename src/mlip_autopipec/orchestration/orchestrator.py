import logging
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult, JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.qe_runner import QERunner
from mlip_autopipec.physics.dynamics.lammps import LammpsResult, LammpsRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
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

        # Setup infrastructure
        # Data directory for accumulated dataset
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.dataset_manager = DatasetManager(self.data_dir)

        # Track current potential
        self.current_potential_path: Optional[Path] = None
        if self.config.training and self.config.training.initial_potential:
            self.current_potential_path = self.config.training.initial_potential

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
            explore_result = self.explore(iter_dir)
            last_result = explore_result

            # 2. Detect
            if self.detect(explore_result):
                logger.info("High uncertainty detected. Proceeding to Refinement.")

                # 3. Select
                candidates = self.select(explore_result)
                logger.info(f"Selected {len(candidates)} candidates for DFT.")

                if not candidates:
                    logger.warning(
                        "No candidates selected despite detection. Convergence?"
                    )
                    break

                # 4. Refine
                try:
                    new_potential = self.refine(candidates, iter_dir)
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
                        log_content=str(e),
                    )
                    break
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

    def explore(self, iter_dir: Path) -> LammpsResult:
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
        md_config = self.config.md

        # Inject uncertainty threshold from OrchestratorConfig
        md_config.uncertainty_threshold = self.config.orchestrator.uncertainty_threshold

        work_dir = iter_dir / "md_run"
        runner = LammpsRunner(
            self.config.lammps, self.config.potential, base_work_dir=work_dir.parent
        )

        # We need to ensure runner uses specific work_dir or we let it create one inside base?
        # LammpsRunner creates a subfolder "job_...".
        # Ideally we want it in `iter_dir/md_run`.
        # LammpsRunner.__init__ takes `base_work_dir`.
        # So passing `iter_dir` as base means it will create `iter_dir/job_...`.
        # For this implementation, we accept that LammpsRunner manages its own job folders.
        runner.base_work_dir = iter_dir  # Override base dir for this run

        result = runner.run(
            structure, md_config, potential_path=self.current_potential_path
        )
        return result

    def detect(self, result: LammpsResult) -> bool:
        """
        Phase 2: Detection
        Analyze the job result for high uncertainty or errors.
        Returns True if retraining is needed (uncertainty detected).
        """
        logger.info("Phase: Detection")

        # Check if result has max_gamma
        if result.max_gamma is not None:
            threshold = self.config.orchestrator.uncertainty_threshold
            logger.info(f"Max Gamma: {result.max_gamma} (Threshold: {threshold})")
            if result.max_gamma > threshold:
                return True

        return False

    def select(self, result: LammpsResult) -> list[Structure]:
        """
        Phase 3: Selection
        Select structures for DFT calculation.
        """
        logger.info("Phase: Selection")

        if not result.trajectory_path.exists():
            logger.warning("No trajectory found.")
            return []

        # Read dump using iread for streaming (memory safety)
        # We expect c_pace_gamma to be available if UQ was on
        traj_iter = ase.io.iread(result.trajectory_path, index=":", format="lammps-dump-text")

        candidates = []
        threshold = self.config.orchestrator.uncertainty_threshold

        # Simple strategy: Select frames where max(gamma) > threshold
        # We process frame by frame to avoid OOM
        # But we also limit the number of candidates.

        for atoms in traj_iter:
            # Check for gamma
            # atoms.arrays might contain 'c_pace_gamma'
            gammas = atoms.arrays.get("c_pace_gamma")

            is_candidate = False
            if gammas is not None:
                if np.max(gammas) > threshold:
                    is_candidate = True
            else:
                # If gamma not present but we detected via log/halt?
                pass

            if is_candidate:
                candidates.append(Structure.from_ase(atoms))

        # If no candidates found via gamma (maybe format issue), but we halted:
        if not candidates and result.status != JobStatus.COMPLETED:
            # Take the final structure
            candidates.append(result.final_structure)

        # Active Set Optimization on candidates?
        # Usually done during training, but we can do it here to reduce DFT calls.
        # But pace_activeset requires a potential to evaluate? No, it selects based on geometric descriptors.
        # But we need to convert to dataset first.
        # For simplicity, we just return candidates, max capped.

        max_size = self.config.orchestrator.max_active_set_size
        if len(candidates) > max_size:
            # Random subsample? Or take spaced frames?
            # Let's just slice
            indices = np.linspace(0, len(candidates) - 1, max_size, dtype=int)
            candidates = [candidates[i] for i in indices]

        return candidates

    def refine(self, candidates: list[Structure], iter_dir: Path) -> Path:
        """
        Phase 4: Refinement
        Run DFT and retrain potential.
        """
        logger.info("Phase: Refinement")

        if not self.config.dft:
            raise ValueError("DFT configuration missing for Refinement phase.")
        if not self.config.training:
            raise ValueError("Training configuration missing for Refinement phase.")

        # 1. DFT (Oracle)
        dft_dir = iter_dir / "dft_calc"
        dft_dir.mkdir(exist_ok=True)

        runner = QERunner(base_work_dir=dft_dir)
        dft_results = []

        for i, s in enumerate(candidates):
            # We skip optimization here (static calc), assuming structure is from MD snapshot
            res = runner.run(s, self.config.dft)
            if res.status == JobStatus.COMPLETED:
                # Update Structure with DFT properties
                # We need to construct a new Structure with properties
                s_new = s.model_copy()
                s_new.properties = {
                    "energy": res.energy,
                    "forces": res.forces,
                    "stress": res.stress,
                }
                dft_results.append(s_new)
            else:
                logger.warning(
                    f"DFT failed for candidate {i}: {res.log_content[:100]}..."
                )

        if not dft_results:
            raise RuntimeError("All DFT calculations failed.")

        # 2. Update Dataset
        # We append to the global accumulated dataset
        dataset_path = self.data_dir / "accumulated.pckl.gzip"
        self.dataset_manager.convert(dft_results, output_path=dataset_path, append=True)

        # 3. Train
        train_dir = iter_dir / "training"
        pacemaker = PacemakerRunner(
            train_dir, self.config.training, self.config.potential
        )

        # Set initial potential for fine-tuning
        # (PacemakerRunner handles config.initial_potential, but we want to update it to current_potential)
        if self.current_potential_path:
            # We need to override the config passed to PacemakerRunner or update it
            # PacemakerRunner uses self.config.initial_potential.
            # We should update the config object passed to it?
            # Or construct a new TrainingConfig
            new_train_config = self.config.training.model_copy()
            new_train_config.initial_potential = self.current_potential_path
            pacemaker.config = new_train_config

        train_result = pacemaker.train(dataset_path)

        if train_result.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Training failed: {train_result.log_content}")

        return train_result.potential_path
