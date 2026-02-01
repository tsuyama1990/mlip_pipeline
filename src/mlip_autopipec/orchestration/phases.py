import logging
import ase.io
import numpy as np
from pathlib import Path
from typing import Optional, List

from mlip_autopipec.domain_models.config import Config, BulkStructureGenConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.qe_runner import QERunner
from mlip_autopipec.physics.dynamics.lammps import LammpsResult, LammpsRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.physics.validation.runner import ValidationRunner

logger = logging.getLogger("mlip_autopipec")


class PhaseExploration:
    """
    Phase 1: Exploration
    Generates structures and runs dynamics to explore the PES.
    """
    def __init__(self, config: Config):
        self.config = config

    def execute(self, iter_dir: Path, potential_path: Optional[Path]) -> LammpsResult:
        logger.info("Phase: Exploration (Structure Gen + MD)")

        # 1. Structure Generation
        gen_config = self.config.structure_gen
        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)
        logger.info(f"Generated structure: {structure.get_chemical_formula()}")

        # 2. Dynamics (MD)
        md_config = self.config.md.model_copy()

        # Inject uncertainty threshold from OrchestratorConfig
        md_config.uncertainty_threshold = self.config.orchestrator.uncertainty_threshold

        work_dir = iter_dir / "md_run"

        # We need a LammpsRunner.
        # Ideally passed in or created. Creating here for now to keep it self-contained
        # but using the config from self.config.
        runner = LammpsRunner(
            config=self.config.lammps,
            potential_config=self.config.potential,
            base_work_dir=work_dir.parent
        )
        # Force work_dir to be inside iteration dir (LammpsRunner creates subdirs by default,
        # but here we want strict control or we let LammpsRunner manage it?)
        # Orchestrator used: runner.base_work_dir = iter_dir
        runner.base_work_dir = iter_dir

        result = runner.run(structure, md_config, potential_path=potential_path)
        return result


class PhaseDetection:
    """
    Phase 2: Detection
    Analyze the job result for high uncertainty or errors.
    """
    def __init__(self, config: Config):
        self.config = config

    def execute(self, result: LammpsResult) -> bool:
        logger.info("Phase: Detection")

        if result.max_gamma is not None:
            threshold = self.config.orchestrator.uncertainty_threshold
            logger.info(f"Max Gamma: {result.max_gamma} (Threshold: {threshold})")
            if result.max_gamma > threshold:
                return True

        # If job failed/halted but we don't have explicit gamma (e.g. error hard),
        # we might assume it's interesting if it wasn't a timeout/system error.
        # But for now, strict gamma check or result status check?
        # Orchestrator logic was: if max_gamma > threshold return True.

        return False


class PhaseSelection:
    """
    Phase 3: Selection
    Select structures for DFT calculation.
    """
    def __init__(self, config: Config):
        self.config = config

    def execute(self, result: LammpsResult) -> List[Structure]:
        logger.info("Phase: Selection")

        if not result.trajectory_path.exists():
            logger.warning("No trajectory found.")
            return []

        # Stream dump using iread
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

        # Fallback: if halted but no specific high-gamma frame found (maybe just the last one triggered it)
        if not candidates and result.status != JobStatus.COMPLETED:
             logger.info("No high-gamma frames found in trajectory, but job halted. Selecting final frame.")
             candidates.append(result.final_structure)

        # Subsampling
        # Using the NEW config field name: max_candidate_pool_size
        max_size = self.config.orchestrator.max_candidate_pool_size
        if len(candidates) > max_size:
            logger.info(f"Subsampling candidates from {len(candidates)} to {max_size}")
            indices = np.linspace(0, len(candidates)-1, max_size, dtype=int)
            candidates = [candidates[i] for i in indices]

        return candidates


class PhaseRefinement:
    """
    Phase 4: Refinement
    Run DFT and retrain potential.
    """
    def __init__(self, config: Config, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.dataset_manager = DatasetManager(self.data_dir)

    def apply_periodic_embedding(self, structure: Structure) -> Structure:
        """
        Apply periodic embedding logic.
        Placeholder for now.
        """
        return structure

    def execute(self, candidates: List[Structure], iter_dir: Path, current_potential_path: Optional[Path]) -> Path:
        logger.info("Phase: Refinement")

        if not self.config.dft:
            raise ValueError("DFT configuration missing.")
        if not self.config.training:
            raise ValueError("Training configuration missing.")

        # 1. DFT (Oracle)
        dft_dir = iter_dir / "dft_calc"
        dft_dir.mkdir(parents=True, exist_ok=True)

        runner = QERunner(base_work_dir=dft_dir)
        dft_results = []

        dataset_path = self.data_dir / "accumulated.pckl.gzip"
        batch_size = self.config.orchestrator.dft_batch_size
        pending_writes = []

        for i, s in enumerate(candidates):
            try:
                s_embedded = self.apply_periodic_embedding(s)
                res = runner.run(s_embedded, self.config.dft)

                if res.status == JobStatus.COMPLETED:
                    s_new = s_embedded.model_copy()

                    # Force Masking
                    forces = res.forces
                    if "ghost_mask" in s_new.arrays:
                        mask = s_new.arrays["ghost_mask"]
                        ghost_indices = np.where(mask)[0]
                        if len(ghost_indices) > 0:
                            forces[ghost_indices] = 0.0

                    s_new.properties = {
                        'energy': res.energy,
                        'forces': forces,
                        'stress': res.stress
                    }
                    dft_results.append(s_new)
                    pending_writes.append(s_new)

                    if len(pending_writes) >= batch_size:
                        self.dataset_manager.convert(pending_writes, output_path=dataset_path, append=True)
                        pending_writes = []
                else:
                    logger.warning(f"DFT failed for candidate {i}: {res.log_content[:100]}...")
            except Exception as e:
                logger.warning(f"DFT execution error for candidate {i}: {e}")

        if pending_writes:
            self.dataset_manager.convert(pending_writes, output_path=dataset_path, append=True)

        if not dft_results:
             raise RuntimeError("All DFT calculations failed.")

        # 3. Train
        train_dir = iter_dir / "training"

        # Instantiate Runner
        pacemaker = PacemakerRunner(
            work_dir=train_dir,
            train_config=self.config.training,
            potential_config=self.config.potential
        )

        # Set initial potential for fine-tuning
        if current_potential_path:
             # Create a copy of training config to inject initial_potential
             new_train_config = self.config.training.model_copy()
             new_train_config.initial_potential = current_potential_path
             pacemaker.config = new_train_config
             pacemaker.train_config = new_train_config # Ensure both are set due to aliasing in Runner

        train_result = pacemaker.train(dataset_path)

        if train_result.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Training failed: {train_result.log_content}")

        if not train_result.potential_path:
             raise RuntimeError("Training completed but no potential path returned.")

        return train_result.potential_path


class PhaseValidation:
    """
    Phase 5: Validation
    Run validation suite.
    """
    def __init__(self, config: Config):
        self.config = config

    def execute(self, potential_path: Path) -> None:
        logger.info("Phase: Validation")
        if not potential_path.exists():
            logger.warning("No potential to validate.")
            return

        # 1. Validation Structure
        gen_config = self.config.structure_gen

        if isinstance(gen_config, BulkStructureGenConfig):
            gen_config = gen_config.model_copy(update={"rattle_stdev": 0.0})

        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)

        # 2. Run Validation
        runner = ValidationRunner(
            val_config=self.config.validation,
            pot_config=self.config.potential,
            potential_path=potential_path
        )
        result = runner.validate(structure)

        logger.info(f"Validation Status: {result.overall_status}")
        for m in result.metrics:
            logger.info(f"Metric {m.name}: {'PASS' if m.passed else 'FAIL'} - {m.value}")
