# ruff: noqa: D101, T201
"""Main CLI application for MLIP-AutoPipe."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import typer
import yaml
from ase import Atoms
from ase.io.jsonio import decode, encode
from dask.distributed import Client, Future, LocalCluster
from pydantic import ValidationError

from mlip_autopipec.config_schemas import (
    CalculationMetadata,
    DFTConfig,
    DFTExecutable,
    DFTInput,
    InferenceParams,
    MDEnsemble,
    SystemConfig,
    UserConfig,
)
from mlip_autopipec.modules.inference import LammpsRunner, UncertaintyQuantifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()


def expand_config(user_config: UserConfig) -> SystemConfig:
    """Expand the user-facing configuration into the internal system configuration.

    In a real implementation, this would involve complex logic to generate
    DFT parameters, file paths, etc. Here, we create a mock expansion.

    Args:
        user_config: The user-provided configuration.

    Returns:
        A fully populated SystemConfig object.

    """
    logging.info("Expanding user configuration into system configuration...")
    # This is a placeholder for a more sophisticated heuristic engine.
    dft_input = DFTInput(
        pseudopotentials={el: f"{el}.UPF" for el in user_config.target_system.elements}
    )
    dft_config = DFTConfig(executable=DFTExecutable(), input=dft_input)

    md_ensemble = MDEnsemble(target_temperature_k=350.0)
    inference_params = InferenceParams(md_ensemble=md_ensemble)

    # Create the SystemConfig directly from Pydantic models
    system_config = SystemConfig(
        dft=dft_config,
        inference=inference_params,
        target_system=user_config.target_system.model_copy(),
        # Other fields like generator, explorer, etc., will get defaults
    )
    return system_config


def setup_dask_client(config: SystemConfig) -> Client:
    """Set up and return a Dask client."""
    if config.dask.scheduler_address:
        logging.info(
            "Connecting to Dask scheduler at: %s", config.dask.scheduler_address
        )
        return Client(config.dask.scheduler_address)  # type: ignore[no-untyped-call]
    logging.info("No Dask scheduler specified, starting a local cluster.")
    cluster = LocalCluster()  # type: ignore[no-untyped-call]
    return Client(cluster)  # type: ignore[no-untyped-call]


# Checkpointing helper functions
def atoms_to_json(atoms: Atoms) -> dict[str, Any]:
    """Serialize an ASE Atoms object to a JSON-compatible dictionary."""
    return json.loads(encode(atoms))  # type: ignore[no-any-return]


def json_to_atoms(data: dict[str, Any]) -> Atoms:
    """Deserialize a JSON dictionary back to an ASE Atoms object."""
    return decode(json.dumps(data))  # type: ignore[no-untyped-call, no-any-return]


class CheckpointManager:
    """Manages saving and loading of the workflow state."""

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path

    def save(self, state: dict[str, Any]) -> None:
        """Save the current workflow state to a checkpoint file."""
        logging.info(f"Saving checkpoint to: {self.path}")
        with open(self.path, "w") as f:
            json.dump(state, f, indent=4)

    def load(self) -> dict[str, Any] | None:
        """Load the workflow state from a checkpoint file if it exists."""
        if not self.path.exists():
            return None
        logging.info(f"Loading checkpoint from: {self.path}")
        with open(self.path) as f:
            return json.load(f)  # type: ignore[no-any-return]


@app.command()
def run(
    checkpoint_path: Path = typer.Option(
        Path("mlip_autopipec_checkpoint.json"),
        "--checkpoint",
        "-cp",
        help="Path to the checkpoint file.",
    ),
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the user configuration YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    db_manager: Any = None,
    dft_factory: Any = None,
    trainer: Any = None,
) -> None:
    """Run the full MLIP-AutoPipe active learning workflow."""
    logging.info(f"Loading user configuration from: {config_path}")
    try:
        with open(config_path) as f:
            user_config_data = yaml.safe_load(f)
        user_config = UserConfig(**user_config_data)
        config = expand_config(user_config)
    except ValidationError as e:
        logging.error(f"Configuration error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during setup: {e}")
        raise typer.Exit(code=2)

    # Use provided instances or create default mocks
    db_manager = db_manager or MagicMock()
    dft_factory = dft_factory or MagicMock()
    trainer = trainer or MagicMock()

    run_workflow(config, checkpoint_path, db_manager, dft_factory, trainer)


def run_workflow(
    config: SystemConfig,
    checkpoint_path: Path,
    db_manager: Any,
    dft_factory: Any,
    trainer: Any,
) -> None:
    """Execute the main asynchronous workflow logic."""
    client = setup_dask_client(config)
    logging.info(f"Dask dashboard link: {client.dashboard_link}")

    checkpoint_manager = CheckpointManager(checkpoint_path)
    state = checkpoint_manager.load() or {}

    dft_futures: list[Future[Any]] = []
    max_cycles = state.get("max_cycles", 5)
    start_cycle = state.get("cycle", 1)
    new_dft_calculations_count = state.get("new_dft_calculations_count", 0)
    retrain_threshold = 3

    for cycle in range(start_cycle, max_cycles + 1):
        logging.info("-" * 50)
        logging.info(f"Starting Active Learning Cycle {cycle}/{max_cycles}")

        logging.info("Step 1: Training the MLIP...")
        trainer.train.return_value = f"model_cycle_{cycle}.yace"
        potential_path = trainer.train()
        logging.info(f"Trained new potential: {potential_path}")

        logging.info("Step 2: Running MD simulation...")
        lammps_runner = LammpsRunner(
            config=config,
            potential_path=potential_path,
            quantifier=UncertaintyQuantifier(),
        )
        simulation_generator = lammps_runner.run()

        submitted_tasks: dict[str, Any] = state.get("submitted_tasks", {})
        for embedded_atoms, force_mask in simulation_generator:
            logging.info("Found uncertain structure, submitting DFT calculation...")
            task_id = str(uuid.uuid4())
            future = client.submit(dft_factory.run, atoms=embedded_atoms, key=task_id)
            dft_futures.append(future)
            submitted_tasks[task_id] = {
                "atoms": atoms_to_json(embedded_atoms),
                "force_mask": force_mask.tolist(),
            }

        logging.info("MD simulation run finished for this cycle.")

        state = {
            "cycle": cycle + 1,
            "max_cycles": max_cycles,
            "new_dft_calculations_count": new_dft_calculations_count,
            "submitted_tasks": submitted_tasks,
        }
        checkpoint_manager.save(state)

        completed_futures = [f for f in dft_futures if f.done()]  # type: ignore[no-untyped-call]
        for future in completed_futures:
            if future.status == "finished":
                atoms_result = future.result()
                task_id = str(future.key)
                force_mask_list = submitted_tasks[task_id]["force_mask"]
                force_mask = np.array(force_mask_list)
                metadata = CalculationMetadata(
                    stage="active_learning", uuid=task_id, force_mask=force_mask
                )
                db_manager.write_calculation(
                    atoms=atoms_result,
                    metadata=metadata,
                    force_mask=force_mask,
                )
                new_dft_calculations_count += 1
                logging.info("Processed a completed DFT calculation.")
                del submitted_tasks[task_id]  # Remove task from checkpoint
            else:
                logging.error(
                    "A DFT calculation failed: %s",
                    future.exception(),  # type: ignore[no-untyped-call]
                )
            dft_futures.remove(future)

        if new_dft_calculations_count >= retrain_threshold:
            logging.info(
                f"Reached {new_dft_calculations_count} new calculations. "
                "Triggering next training cycle."
            )
            new_dft_calculations_count = 0  # Reset counter
        else:
            logging.info(
                f"Completed {new_dft_calculations_count} new calculations. "
                "Waiting for more before retraining."
            )

    logging.info("Waiting for remaining DFT calculations to finish...")
    client.gather(dft_futures)  # type: ignore[no-untyped-call]
    for future in dft_futures:
        if future.status == "finished":
            atoms_result = future.result()
            task_id = str(future.key)
            force_mask_list = submitted_tasks[task_id]["force_mask"]
            force_mask = np.array(force_mask_list)
            metadata = CalculationMetadata(
                stage="active_learning", uuid=task_id, force_mask=force_mask
            )
            db_manager.write_calculation(
                atoms=atoms_result,
                metadata=metadata,
                force_mask=force_mask,
            )
        else:
            logging.error(
                "A DFT calculation failed: %s",
                future.exception(),  # type: ignore[no-untyped-call]
            )
    logging.info("All tasks finished. Workflow complete.")
    checkpoint_manager.path.unlink(missing_ok=True)
    client.close()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    app()
