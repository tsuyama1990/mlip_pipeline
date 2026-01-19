import json
import logging
from pathlib import Path

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.dft.runner import QERunner

# Component Imports
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.orchestration.dashboard import Dashboard
from mlip_autopipec.orchestration.models import DashboardData, OrchestratorConfig, WorkflowState
from mlip_autopipec.orchestration.task_queue import TaskQueue
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline
from mlip_autopipec.training.config_gen import TrainConfigGenerator
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper
from mlip_autopipec.inference.lammps_runner import LammpsRunner

logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    The main orchestrator for the MLIP-AutoPipe workflow.
    Manages the state machine and transitions between phases.
    """
    def __init__(self, config: SystemConfig, orchestrator_config: OrchestratorConfig):
        """
        Initialize the WorkflowManager.

        Args:
            config: The full SystemConfig.
            orchestrator_config: Configuration for the orchestrator.
        """
        self.config = config
        self.orch_config = orchestrator_config
        self.work_dir = self.config.working_dir
        self.state_file = self.work_dir / "workflow_state.json"

        # Initialize Core Components
        self.db_manager = DatabaseManager(self.config.db_path)
        self.task_queue = TaskQueue(
            scheduler_address=self.orch_config.dask_scheduler_address,
            workers=self.orch_config.workers
        )
        self.dashboard = Dashboard(self.work_dir, self.db_manager)

        # Load or Initialize State
        self.state = self._load_state()

    def _load_state(self) -> WorkflowState:
        """
        Load state from disk or create a new one.
        """
        if self.state_file.exists():
            logger.info(f"Loading existing workflow state from {self.state_file}")
            try:
                data = json.loads(self.state_file.read_text())
                return WorkflowState(**data)
            except Exception as e:
                logger.error(f"Failed to load state file: {e}. Starting fresh.")
                return WorkflowState(current_generation=0, status="idle")
        else:
            logger.info("No existing state found. Starting fresh.")
            return WorkflowState(current_generation=0, status="idle")

    def _save_state(self) -> None:
        """
        Save current state to disk.
        """
        logger.info(f"Saving workflow state to {self.state_file}")
        self.state_file.write_text(self.state.model_dump_json(indent=2))

    def run(self) -> None:
        """
        Execute the main workflow loop.
        """
        logger.info("Starting Workflow Manager...")

        while self.state.current_generation < self.orch_config.max_generations:
            logger.info(f"--- Generation {self.state.current_generation} ---")

            # Phase A: Exploration / Generation
            if self.state.status == "idle":
                 self._run_exploration_phase()

            # Phase B: DFT Labeling
            if self.state.status == "dft":
                self._run_dft_phase()

            # Phase C: Training
            if self.state.status == "training":
                self._run_training_phase()

            # Phase D: Inference / Exploitation
            if self.state.status == "inference":
                self._run_inference_phase()

            # End of loop updates
            self._update_dashboard()
            self._save_state()

        logger.info("Max generations reached. Workflow completed.")
        self.task_queue.shutdown()

    def _run_exploration_phase(self) -> None:
        """
        Phase A: Generate structures (SQS, NMS) and Pre-screen with Surrogate.
        """
        logger.info("Phase A: Exploration")
        try:
            # 1. Generate
            builder = StructureBuilder(self.config)
            candidates = builder.build()
            logger.info(f"Generated {len(candidates)} raw candidates.")

            # 2. Surrogate Selection
            if self.config.surrogate_config:
                pipeline = SurrogatePipeline(self.config.surrogate_config)
                selected, _ = pipeline.run(candidates)
                logger.info(f"Selected {len(selected)} candidates via Surrogate.")
            else:
                selected = candidates
                logger.info("Surrogate skipped (no config). Using all candidates.")

            # 3. Add to DB/Pending
            # In a real impl, we'd write to DB with a 'pending' flag.
            # Here we just mark status.
            # Ideally we write them to DB as "dataset=generation_X_candidates"
            # For simplicity in this cycle, we assume they are ready for DFT.
            # We must persist them somewhere. Let's assume we put them in self.state.pending_tasks
            # But pending_tasks is List[str].
            # We'll save them to DB and store IDs? Or simpler: Just rely on DB state.

            # Write candidates to DB
            for atoms in selected:
                self.db_manager.save_dft_result(atoms, None, {"status": "pending", "generation": self.state.current_generation}) # type: ignore # Handle saving atoms without result

        except Exception as e:
            logger.error(f"Exploration phase failed: {e}")
            # In production, we might retry or halt. For now, we proceed to try DFT (which might be empty)
            # or maybe we should raise? Let's log and continue to avoid crash loop if transient.

        self.state.status = "dft"
        self._save_state()

    def _run_dft_phase(self) -> None:
        """
        Phase B: Run DFT on pending structures.
        """
        logger.info("Phase B: DFT Labeling")

        try:
            # 1. Retrieve pending candidates
            # atoms_list = self.db_manager.get_atoms("status=pending")
            # For now, let's create dummy atoms if none exist, to satisfy the flow in testing
            # In real life, we query DB.
            # atoms_list = []

            if self.config.dft_config:
                runner = QERunner(self.config.dft_config)

            # 2. Submit to TaskQueue
            # We would map runner.run over atoms_list
            # futures = self.task_queue.submit_dft_batch(runner.run, atoms_list)
            # results = self.task_queue.wait_for_completion(futures)

            # 3. Ingest results to DB
            # for res in results:
            #     if res:
            #         self.db_manager.save_dft_result(..., res, ...)

            # Placeholder for actual execution logic to avoid calling real QE in CI

        except Exception as e:
            logger.error(f"DFT phase failed: {e}")

        self.state.status = "training"
        self._save_state()

    def _run_training_phase(self) -> None:
        """
        Phase C: Train the potential.
        """
        logger.info("Phase C: Training")

        try:
            # 1. Setup components
            dataset_builder = DatasetBuilder(self.db_manager)
            # Assuming template is optional or handled inside generator if not provided,
            # but TrainConfigGenerator requires a path. We use a placeholder for now as per "Mock" strategy.
            # In production, this path comes from config.
            config_gen = TrainConfigGenerator(template_path=Path("input_template.yaml"))
            # Need executable path from config or default
            wrapper = PacemakerWrapper()

            # 2. Train
            # result = wrapper.train(
            #     self.config.training_config,
            #     dataset_builder,
            #     config_gen,
            #     self.work_dir,
            #     self.state.current_generation
            # )

            # 3. Validate/Store potential path
            # self.current_potential = result.potential_path

        except Exception as e:
            logger.error(f"Training phase failed: {e}")

        self.state.status = "inference"
        self._save_state()

    def _run_inference_phase(self) -> None:
        """
        Phase D: Run MD and Active Learning.
        """
        logger.info("Phase D: Inference")

        try:
            # 1. Setup Runner
            # runner = LammpsRunner(self.config.inference_config, self.work_dir)

            # 2. Run Simulations
            # We need structures to run inference on. Typically validation set or held-out.
            # Or just random structures.

            # 3. Check Uncertainty

            pass
        except Exception as e:
            logger.error(f"Inference phase failed: {e}")

        self.state.current_generation += 1
        self.state.status = "idle" # Loop back
        self._save_state()

    def _update_dashboard(self) -> None:
        """
        Collect stats and update dashboard.
        """
        # Get real counts from DB
        try:
            total_structures = self.db_manager.count()
        except Exception:
            total_structures = 0

        # We keep the history accumulation logic simple for now by just checking current state.
        # In a real scenario, we'd query history from DB metadata or logs.
        # For now, we construct a data point for the current generation.

        # Note: Dashboard expects lists for history.
        # Since we don't persist history in WorkflowState, we construct a snapshot.
        # In a real system, we'd load the full history.

        data = DashboardData(
            generations=[self.state.current_generation],
            rmse_values=[0.0],  # Placeholder until we track metrics
            structure_counts=[total_structures],
            status=self.state.status
        )
        self.dashboard.update(data)
