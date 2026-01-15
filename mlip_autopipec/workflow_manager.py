# ruff: noqa: D101, T201
"""Module for the main workflow orchestration."""

import logging
from pathlib import Path
from typing import Any

from dask.distributed import Client, Future

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.inference import LammpsRunner, UncertaintyQuantifier
from mlip_autopipec.utils.workflow_utils import (
    CheckpointManager,
    atoms_to_json,
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
    ):
        self.config = config
        self.checkpoint_manager = CheckpointManager(checkpoint_path)
        self.db_manager = db_manager
        self.dft_factory = dft_factory
        self.trainer = trainer
        self.client = client

    def run(self) -> None:
        """Execute the main asynchronous workflow logic."""
        logging.info(f"Dask dashboard link: {self.client.dashboard_link}")
        state = self.checkpoint_manager.load() or {}

        dft_futures: list[Future[Any]] = []
        max_cycles = state.get("max_cycles", 5)
        start_cycle = state.get("cycle", 1)
        new_dft_calculations_count = state.get("new_dft_calculations_count", 0)
        retrain_threshold = 3

        for cycle in range(start_cycle, max_cycles + 1):
            logging.info("-" * 50)
            logging.info(f"Starting Active Learning Cycle {cycle}/{max_cycles}")

            self._run_training_and_md(state, dft_futures)

            new_dft_calculations_count = self._process_completed_dft_futures(
                dft_futures, new_dft_calculations_count, state
            )

            if new_dft_calculations_count >= retrain_threshold:
                logging.info(
                    f"Reached {new_dft_calculations_count} new calculations. "
                    "Triggering next training cycle."
                )
                new_dft_calculations_count = 0  # Reset counter

        self._wait_for_remaining_dft_futures(dft_futures, state)
        self.client.close()  # type: ignore[no-untyped-call]

    def _run_training_and_md(
        self, state: dict[str, Any], dft_futures: list[Future[Any]]
    ) -> None:
        """Run the training and MD simulation for a single cycle."""
        logging.info("Step 1: Training the MLIP...")
        self.trainer.train.return_value = "model.yace"
        potential_path = self.trainer.train()
        logging.info(f"Trained new potential: {potential_path}")

        logging.info("Step 2: Running MD simulation...")
        lammps_runner = LammpsRunner(
            config=self.config,
            potential_path=potential_path,
            quantifier=UncertaintyQuantifier(),
        )
        simulation_generator = lammps_runner.run()
        submitted_tasks: dict[str, Any] = state.get("submitted_tasks", {})

        for embedded_atoms, force_mask in simulation_generator:
            logging.info("Found uncertain structure, submitting DFT calculation...")
            future = self.client.submit(self.dft_factory.run, embedded_atoms)
            dft_futures.append(future)
            submitted_tasks[str(future.key)] = {
                "atoms": atoms_to_json(embedded_atoms),
                "force_mask": force_mask.tolist(),
            }
        state["submitted_tasks"] = submitted_tasks

    def _process_completed_dft_futures(
        self,
        dft_futures: list[Future[Any]],
        new_dft_calculations_count: int,
        state: dict[str, Any],
    ) -> int:
        """Process completed DFT calculations and update the database."""
        completed_futures = [f for f in dft_futures if f.done()]  # type: ignore[no-untyped-call]
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
        self.client.gather(dft_futures)  # type: ignore[no-untyped-call]
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
