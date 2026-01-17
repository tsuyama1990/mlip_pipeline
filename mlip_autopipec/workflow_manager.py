import json
import logging
from pathlib import Path
from uuid import UUID

from dask.distributed import Client, Future, as_completed
from pydantic import ValidationError

from mlip_autopipec.config.models import CheckpointState, SystemConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.data_models.training_data import TrainingBatch
from mlip_autopipec.modules.dft import DFTRunner
from mlip_autopipec.modules.training import PacemakerTrainer
from mlip_autopipec.utils.dask_utils import get_dask_client

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    The central orchestrator for the MLIP-AutoPipe workflow.

    This class is responsible for initializing and coordinating the various
    modules, managing the main active learning loop with Dask for parallel
    execution, and handling checkpointing for resilience.
    """

    def __init__(
        self,
        system_config: SystemConfig,
        work_dir: Path,
        dft_runner: DFTRunner | None = None,
        trainer: PacemakerTrainer | None = None,
    ) -> None:
        self.system_config = system_config
        self.work_dir = work_dir
        self.checkpoint_path = self.work_dir / system_config.workflow_config.checkpoint_file_path
        self.dask_client: Client = get_dask_client()
        self.dft_runner: DFTRunner | None = dft_runner
        self.trainer: PacemakerTrainer | None = trainer
        self.futures: dict[UUID, Future] = {}
        self.state: CheckpointState | None = None

        # Initialize DatabaseManager using the config path
        # In a real scenario, db_path in config should be absolute or resolved.
        # Here we trust the SystemConfig factory.
        self.db_manager = DatabaseManager(system_config.db_path)

        self._load_or_initialize_state()

    def _load_or_initialize_state(self):
        """Loads state from a checkpoint or initializes a new one."""
        if self.checkpoint_path.exists():
            self._load_checkpoint()
            self._resubmit_pending_jobs()
        else:
            self.state = CheckpointState(
                run_uuid=self.system_config.run_uuid,
                system_config=self.system_config,
            )
            self._save_checkpoint()
            logger.info("Initialized a new workflow state.")

    def _save_checkpoint(self):
        """Serializes the current state to a checkpoint file."""
        logger.info("Saving checkpoint...")
        with self.checkpoint_path.open("w") as f:
            # Pydantic's model_dump_json handles serialization of complex types
            # like UUID and Path to standard JSON types.
            f.write(self.state.model_dump_json(indent=4))
        logger.info("Checkpoint saved to %s", self.checkpoint_path)

    def _load_checkpoint(self):
        """Loads and validates the state from a checkpoint file."""
        logger.info("Loading state from checkpoint: %s", self.checkpoint_path)
        try:
            with self.checkpoint_path.open() as f:
                state_data = json.load(f)
                self.state = CheckpointState.model_validate(state_data)
            logger.info("Successfully loaded workflow state.")
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            logger.error("Failed to load or validate checkpoint.", exc_info=True)
            msg = "Could not load a valid checkpoint file."
            raise RuntimeError(msg) from e

    def _resubmit_pending_jobs(self):
        """Re-submits jobs that were pending at the time of the last checkpoint."""
        if not self.state.pending_job_ids:
            return

        logger.info("Re-submitting %d pending jobs...", len(self.state.pending_job_ids))
        if self.dft_runner is None:
            # This is a safeguard; dft_runner should be set before this is called in a real run.
            msg = "DFTRunner is not initialized."
            raise RuntimeError(msg)

        for job_id in self.state.pending_job_ids:
            args = self.state.job_submission_args.get(job_id)
            if args:
                future = self.dask_client.submit(self.dft_runner.run, *args)
                self.futures[job_id] = future
            else:
                logger.warning("No submission arguments found for pending job ID: %s", job_id)
        logger.info("All pending jobs have been re-submitted.")

    def perform_training(self):
        """Executes the training step and updates state with metrics."""
        if not self.trainer:
            logger.warning("Trainer not initialized. Skipping training.")
            return

        try:
            logger.info(f"Reading training data from {self.system_config.db_path}...")
            # Use DatabaseManager to fetch data
            training_data = self.db_manager.get_training_data()
        except Exception:
            logger.exception("Failed to read training data from database.")
            return  # Don't crash, just skip training this cycle

        if not training_data:
            logger.warning("No training data found in database. Skipping training.")
            return

        logger.info("Starting training for generation %d...", self.state.active_learning_generation)
        try:
            # Wrap raw list in Pydantic model for validation
            batch = TrainingBatch(atoms_list=training_data)

            potential_path, metrics = self.trainer.perform_training(
                training_data=batch, generation=self.state.active_learning_generation
            )

            # Update state
            self.state.training_history.append(metrics)
            self.state.current_potential_path = potential_path

            self._save_checkpoint()
            logger.info("Training completed successfully. Metrics saved.")

        except Exception:
            logger.exception("Training failed.")
            # Depending on policy, we might re-raise or just log.
            # For now re-raise to be safe.
            raise

    def run(self):
        """
        Executes the main active learning workflow.

        This method is a placeholder for the full active learning loop. In this
        cycle, it focuses on demonstrating the Dask integration and checkpointing
        by managing a batch of futures.
        """
        logger.info("WorkflowManager: run() started.")
        # In a full implementation, structures would be generated here and
        # submitted as jobs. For now, we assume jobs are submitted externally
        # for testing purposes.

        if not self.futures:
            logger.info("No active jobs to monitor. Workflow exiting.")
            return

        # Process results as they complete
        for future in as_completed(self.futures.values()):
            try:
                result = future.result()
                job_id = result.job_id

                # Update state: remove job from pending and args dict
                if job_id in self.state.pending_job_ids:
                    self.state.pending_job_ids.remove(job_id)
                if job_id in self.state.job_submission_args:
                    del self.state.job_submission_args[job_id]

                # Save the result to the database (placeholder for actual saving logic)
                # In Cycle 02/08 this would use self.db_manager.save_dft_result(...)
                logger.info("Processed result for job %s", job_id)

                # Save state after processing the result
                self._save_checkpoint()

            except Exception:
                logger.error("A DFT calculation job failed.", exc_info=True)
                # Here you might add logic to handle failed jobs, e.g.,
                # marking them as failed in the state.

        logger.info("All jobs completed. Workflow finished.")

        # Trigger training if configured
        if self.trainer:
            self.perform_training()
