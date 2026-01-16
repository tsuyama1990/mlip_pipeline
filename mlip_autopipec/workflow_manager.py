"""Module for the main workflow orchestration."""

import logging
from pathlib import Path
from typing import Any

from dask.distributed import Client, Future

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.utils.workflow_utils import (
    CheckpointManager,
)


class WorkflowManager:
    """Orchestrates the entire MLIP-AutoPipe workflow."""

    def __init__(
        self,
        config: SystemConfig,
        checkpoint_path: Path,
        db_manager: Any,
        dft_factory: Any,
        trainer: Any,
        client: Client,
    ) -> None:
        self.config = config
        self.checkpoint_manager = CheckpointManager(checkpoint_path)
        self.db_manager = db_manager
        self.dft_factory = dft_factory
        self.trainer = trainer
        self.client = client

    def run(self) -> None:
        """Execute the main asynchronous workflow logic."""
        logging.info("Dask dashboard link: %s", self.client.dashboard_link)
        state = self.checkpoint_manager.load() or {}

        dft_futures: list[Future[Any]] = []
        max_cycles = state.get("max_cycles", 5)
        start_cycle = state.get("cycle", 1)
        new_dft_calculations_count = state.get("new_dft_calculations_count", 0)
        retrain_threshold = 3

        for cycle in range(start_cycle, max_cycles + 1):
            logging.info("-" * 50)
            logging.info("Starting Active Learning Cycle %d/%d", cycle, max_cycles)

            self._run_training_and_md(state, dft_futures)

            new_dft_calculations_count = self._process_completed_dft_futures(
                dft_futures, new_dft_calculations_count, state
            )

            if new_dft_calculations_count >= retrain_threshold:
                logging.info(
                    "Reached %d new calculations. Triggering next training cycle.",
                    new_dft_calculations_count,
                )
                new_dft_calculations_count = 0  # Reset counter

        self._wait_for_remaining_dft_futures(dft_futures, state)
        self.client.close()

    def _run_training_and_md(self, state: dict[str, Any], dft_futures: list[Future[Any]]) -> None:
        """Run the training and MD simulation for a single cycle."""
        logging.info("Step 1: Training the MLIP...")
        self.trainer.train.return_value = "model.yace"
        potential_path = self.trainer.train()
        logging.info("Trained new potential: %s", potential_path)

        logging.info("Step 2: Running MD simulation...")
        # Note: This part of the workflow manager will need to be updated
        # in a future cycle to properly integrate the new InferenceConfig.
        # For now, we are focusing on the implementation of LammpsRunner itself.
        # lammps_runner = LammpsRunner(inference_config=self.config.inference)
        # result = lammps_runner.run(initial_structure)

    def _process_completed_dft_futures(
        self,
        dft_futures: list[Future[Any]],
        new_dft_calculations_count: int,
        state: dict[str, Any],
    ) -> int:
        """Process completed DFT calculations and update the database."""
        completed_futures = [f for f in dft_futures if f.done()]
        for future in completed_futures:
            if future.status == "finished":
                atoms_result = future.result()
                task_id = str(future.key)
                force_mask = state["submitted_tasks"][task_id]["force_mask"]
                self.db_manager.write_calculation(
                    atoms=atoms_result,
                    metadata={"stage": "active_learning", "uuid": task_id},
                    force_mask=force_mask,
                )
                new_dft_calculations_count += 1
                del state["submitted_tasks"][task_id]
            dft_futures.remove(future)
        self.checkpoint_manager.save(state)
        return new_dft_calculations_count

    def _wait_for_remaining_dft_futures(
        self, dft_futures: list[Future[Any]], state: dict[str, Any]
    ) -> None:
        """Wait for and process any remaining DFT calculations."""
        logging.info("Waiting for remaining DFT calculations to finish...")
        self.client.gather(dft_futures)
        for future in dft_futures:
            if future.status == "finished":
                atoms_result = future.result()
                task_id = str(future.key)
                force_mask = state["submitted_tasks"][task_id]["force_mask"]
                self.db_manager.write_calculation(
                    atoms=atoms_result,
                    metadata={"stage": "active_learning", "uuid": task_id},
                    force_mask=force_mask,
                )
        self.checkpoint_manager.path.unlink(missing_ok=True)
