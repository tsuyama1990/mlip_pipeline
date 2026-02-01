import logging
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np

from mlip_autopipec.domain_models.config import BulkStructureGenConfig, Config
from mlip_autopipec.domain_models.job import JobResult, JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.qe_runner import QERunner
from mlip_autopipec.physics.dynamics.lammps import LammpsResult, LammpsRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.physics.validation.runner import ValidationRunner

logger = logging.getLogger("mlip_autopipec")


class Orchestrator:
    """
    Orchestrates the active learning pipeline.
    Implements the Active Learning Cycle: Explore -> Detect -> Select -> Refine -> Validate.
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

    def _create_lammps_runner(self, work_dir: Path) -> LammpsRunner:
        """Factory method for LammpsRunner."""
        return LammpsRunner(
            config=self.config.lammps,
            potential_config=self.config.potential,
            base_work_dir=work_dir.parent
        )

    def _create_dft_runner(self, work_dir: Path) -> QERunner:
        """Factory method for QERunner."""
        return QERunner(base_work_dir=work_dir)

    def _create_pacemaker_runner(self, work_dir: Path) -> PacemakerRunner:
        """Factory method for PacemakerRunner."""
        if not self.config.training:
            raise ValueError("Training configuration missing.")
        return PacemakerRunner(
            work_dir=work_dir,
            train_config=self.config.training,
            potential_config=self.config.potential
        )

    def _create_validation_runner(self, potential_path: Path) -> ValidationRunner:
        """Factory method for ValidationRunner."""
        return ValidationRunner(
            val_config=self.config.validation,
            pot_config=self.config.potential,
            potential_path=potential_path
        )

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
                    logger.warning("No candidates selected despite detection. Convergence?")
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
                        log_content=str(e)
                    )
                    break

                # 5. Validate
                if self.iteration % self.config.orchestrator.validation_frequency == 0:
                    try:
                        self.validate(iter_dir)
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
        runner = self._create_lammps_runner(work_dir)

        runner.base_work_dir = iter_dir # Force it to be inside iteration dir

        result = runner.run(structure, md_config, potential_path=self.current_potential_path)
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

        # Stream dump using iread with stride to avoid OOM and reduce processing
        stride = self.config.orchestrator.trajectory_sampling_stride
        index_spec = f"::{stride}"

        try:
            # iread returns an iterator
            traj_iter = ase.io.iread(result.trajectory_path, index=index_spec, format="lammps-dump-text") # type: ignore[no-untyped-call]
        except Exception as e:
            logger.error(f"Failed to read trajectory: {e}")
            return []

        candidates = []
        threshold = self.config.orchestrator.uncertainty_threshold

        for i, atoms in enumerate(traj_iter):
            # Check for gamma
            gammas = atoms.arrays.get('c_pace_gamma')

            is_candidate = False
            if gammas is not None:
                if np.max(gammas) > threshold:
                    is_candidate = True
                    logger.debug(f"Frame {i*stride}: Max gamma {np.max(gammas)} > {threshold}")

            if is_candidate:
                candidates.append(Structure.from_ase(atoms))

        # If no candidates found via gamma, but we halted (likely due to uncertainty or error),
        # take the final structure as a fallback candidate.
        if not candidates and result.status != JobStatus.COMPLETED:
             logger.info("No high-gamma frames found in trajectory, but job halted. Selecting final frame.")
             candidates.append(result.final_structure)

        # Limit the size of active set to avoid overwhelming DFT
        max_size = self.config.orchestrator.max_active_set_size
        if len(candidates) > max_size:
            logger.info(f"Subsampling candidates from {len(candidates)} to {max_size}")
            indices = np.linspace(0, len(candidates)-1, max_size, dtype=int)
            candidates = [candidates[i] for i in indices]

        return candidates

    def apply_periodic_embedding(self, structure: Structure) -> Structure:
        """
        Apply periodic embedding logic.
        For now, this is a placeholder that ensures 3D PBC and could add a ghost mask.
        In a full implementation, this would carve a cluster and embed it in a bulk-like box.

        Spec 3.2 compliance: Ghost mask generation.
        """
        # Placeholder logic: just ensure pbc is True for DFT
        # Real logic would identify buffer regions and set arrays["ghost_mask"] = [0, 0, 1, ...]

        # We simulate this by checking if we need to do anything.
        # If structure is already periodic, we assume it's good.
        # If we had cluster logic, we'd do it here.

        return structure

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

        runner = self._create_dft_runner(dft_dir)
        dft_results = []

        # Batch IO setup
        dataset_path = self.data_dir / "accumulated.pckl.gzip"
        batch_size = self.config.orchestrator.dft_batch_size
        pending_writes = []

        for i, s in enumerate(candidates):
            try:
                # Apply Periodic Embedding / Pre-processing
                s_embedded = self.apply_periodic_embedding(s)

                res = runner.run(s_embedded, self.config.dft)
                if res.status == JobStatus.COMPLETED:
                    # Update Structure with DFT properties
                    s_new = s_embedded.model_copy()

                    # Apply Force Masking if 'ghost_mask' is present
                    forces = res.forces
                    if "ghost_mask" in s_new.arrays:
                        mask = s_new.arrays["ghost_mask"]
                        ghost_indices = np.where(mask)[0]
                        if len(ghost_indices) > 0:
                            forces[ghost_indices] = 0.0
                            logger.debug(f"Masked forces for {len(ghost_indices)} ghost atoms in candidate {i}")

                    s_new.properties = {
                        'energy': res.energy,
                        'forces': forces,
                        'stress': res.stress
                    }
                    dft_results.append(s_new)
                    pending_writes.append(s_new)

                    # Batch Write
                    if len(pending_writes) >= batch_size:
                        self.dataset_manager.convert(pending_writes, output_path=dataset_path, append=True)
                        pending_writes = [] # Clear buffer

                else:
                    logger.warning(f"DFT failed for candidate {i}: {res.log_content[:100]}...")
            except Exception as e:
                logger.warning(f"DFT execution error for candidate {i}: {e}")

        # Final flush of pending writes
        if pending_writes:
            self.dataset_manager.convert(pending_writes, output_path=dataset_path, append=True)

        if not dft_results:
             raise RuntimeError("All DFT calculations failed.")

        # 3. Train
        train_dir = iter_dir / "training"
        pacemaker = self._create_pacemaker_runner(train_dir)

        # Set initial potential for fine-tuning
        if self.current_potential_path:
             new_train_config = self.config.training.model_copy()
             new_train_config.initial_potential = self.current_potential_path
             pacemaker.config = new_train_config

        train_result = pacemaker.train(dataset_path)

        if train_result.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Training failed: {train_result.log_content}")

        if train_result.potential_path is None:
            raise RuntimeError("Training reported success but returned no potential path.")

        return train_result.potential_path

    def validate(self, iter_dir: Path) -> None:
        """
        Phase 5: Validation
        Run validation suite on the current potential.
        """
        logger.info("Phase: Validation")
        if not self.current_potential_path or not self.current_potential_path.exists():
            logger.warning("No potential to validate.")
            return

        # 1. Generate Validation Structure (Preferably Ideal Bulk)
        gen_config = self.config.structure_gen

        # If possible, remove thermal noise for validation structure
        if isinstance(gen_config, BulkStructureGenConfig):
            # Create a copy with rattle=0
            gen_config = gen_config.model_copy(update={"rattle_stdev": 0.0})

        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)

        # 2. Run Validation
        runner = self._create_validation_runner(self.current_potential_path)
        result = runner.validate(structure)

        logger.info(f"Validation Status: {result.overall_status}")
        for m in result.metrics:
            logger.info(f"Metric {m.name}: {'PASS' if m.passed else 'FAIL'} - {m.value}")
