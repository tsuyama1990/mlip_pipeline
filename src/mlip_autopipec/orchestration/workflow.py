import logging
import uuid
from pathlib import Path

import numpy as np
from rich.progress import track

from mlip_autopipec.config.models import SystemConfig, UserInputConfig
from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.domain_models.dft_models import DFTResult
from mlip_autopipec.domain_models.state import WorkflowState
from mlip_autopipec.generator import StructureBuilder
from mlip_autopipec.modules.training_orchestrator import TrainingManager
from mlip_autopipec.orchestration.database import DatabaseManager

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Orchestrates the active learning cycles.
    """

    def __init__(self, config: UserInputConfig, work_dir: Path, state_file: Path | None = None, workflow_config=None):
        self.config = config
        self.work_dir = work_dir
        self.state_file = state_file or (work_dir / "state.json")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Allow workflow_config override or use from config
        self.workflow_config = workflow_config or config.workflow_config

        # Use config value for db path, not hardcoded
        self.db_path = self.work_dir / self.config.runtime.database_path
        self.db_manager = DatabaseManager(self.db_path)

        self.state = self._load_state()

    def _load_state(self) -> WorkflowState:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return WorkflowState.model_validate_json(f.read())
        return WorkflowState()

    def save_state(self) -> None:
        with open(self.state_file, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def run(self) -> None:
        """Main execution loop for full active learning (Future Cycles)."""
        max_cycles = self.workflow_config.max_generations if self.workflow_config else 5

        while self.state.cycle_index < max_cycles:
            logger.info(f"Starting Cycle {self.state.cycle_index}")
            # Placeholder for full loop: Exploration -> Selection -> DFT -> Training
            self.state.cycle_index += 1
            self.save_state()

    def run_cycle_02(self, mock_dft: bool = False, dry_run: bool = False) -> None:
        """
        Executes Cycle 02 pipeline: Generation -> DFT (Oracle) -> Database -> Training.
        """
        logger.info("Starting Cycle 02 Pipeline")

        # 1. Structure Generation
        logger.info("Step 1: Structure Generation")
        sys_config = SystemConfig(
            target_system=self.config.target_system,
            generator_config=self.config.generator_config
        )
        builder = StructureBuilder(sys_config)
        structures = list(builder.build())
        logger.info(f"Generated {len(structures)} structures.")

        if dry_run:
            logger.info("Dry run: Skipping DFT and Training.")
            return

        # 2. Oracle (DFT) & 3. Database
        logger.info("Step 2: Oracle Calculation & Database Storage")

        runner = None
        if not mock_dft:
            if not self.config.dft:
                msg = "DFT config missing."
                raise ValueError(msg)
            # Use runtime config for DFT work dir
            dft_work_dir = self.work_dir / "dft_runs"
            runner = QERunner(self.config.dft, work_dir=dft_work_dir)
        else:
            logger.warning("Running in MOCK DFT mode.")

        with self.db_manager as db:
            db.set_system_config(sys_config)

            for atoms in track(structures, description="Running DFT..."):
                uid = str(uuid.uuid4())
                atoms.info["uid"] = uid
                atoms.info["generation"] = 0

                result = None
                if mock_dft:
                    # Mock Result
                    energy = -3.5 * len(atoms) + np.random.normal(0, 0.1)
                    forces = np.random.normal(0, 0.05, size=(len(atoms), 3)).tolist()
                    stress = np.zeros((3, 3)).tolist()
                    result = DFTResult(
                        uid=uid,
                        energy=energy,
                        forces=forces,
                        stress=stress,
                        succeeded=True,
                        converged=True,
                        wall_time=0.1,
                        parameters={"mock": True},
                    )
                elif runner:
                    try:
                        result = runner.run(atoms, uid=uid)
                    except Exception:
                        logger.exception(f"DFT Failed for {uid}")
                        continue

                if result and result.succeeded:
                    # Save to DB
                    db.save_dft_result(atoms, result, metadata={"status": "completed", "cycle": 0})
                else:
                    logger.error(f"DFT failed for {uid}")

            # 4. Training
            logger.info("Step 3: Training")

            if not self.config.training_config:
                msg = "Training config missing."
                raise ValueError(msg)

            training_dir = self.work_dir / "training"
            training_dir.mkdir(parents=True, exist_ok=True)

            manager = TrainingManager(db, self.config.training_config, training_dir)
            train_result = manager.run_training()

            if train_result.success:
                logger.info("Training Successful!")
                logger.info(f"Potential: {train_result.potential_path}")
                self.state.latest_potential_path = train_result.potential_path
                self.save_state()
            else:
                logger.error("Training Failed.")
                msg = "Training Failed"
                raise RuntimeError(msg)
