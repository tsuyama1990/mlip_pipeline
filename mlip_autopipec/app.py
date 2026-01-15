# ruff: noqa: D101, T201
"""Main CLI application for MLIP-AutoPipe."""

import logging
import json
from pathlib import Path
import time
import uuid

import dask
import numpy as np
import typer
import yaml
from ase import Atoms
from dask.distributed import Client, as_completed
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
from mlip_autopipec.data.database import DatabaseManager
from mlip_autopipec.modules.dft.dft_factory import (
    QEInputGenerator,
    QEOutputParser,
    QEProcessRunner,
)
from mlip_autopipec.modules.inference import LammpsRunner, UncertaintyQuantifier
from mlip_autopipec.modules.trainer import PacemakerTrainer

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


def dft_task_wrapper(
    config: SystemConfig, atoms: Atoms, force_mask: np.ndarray | None
) -> tuple[Atoms, CalculationMetadata, np.ndarray | None]:
    """A self-contained Dask task for running a DFT calculation."""
    input_gen = QEInputGenerator()
    output_parser = QEOutputParser()
    dft_runner = QEProcessRunner(config, input_gen, output_parser)
    atoms_result = dft_runner.run(atoms)
    metadata = CalculationMetadata(stage="active_learning", uuid=str(uuid.uuid4()))
    return atoms_result, metadata, force_mask


def save_checkpoint(state: dict):
    """Save the application state to a checkpoint file."""
    with open("mlip_autopipec_checkpoint.json", "w") as f:
        json.dump(state, f)
    logging.info("Checkpoint saved.")

def load_checkpoint() -> dict:
    """Load the application state from a checkpoint file."""
    if Path("mlip_autopipec_checkpoint.json").exists():
        with open("mlip_autopipec_checkpoint.json", "r") as f:
            logging.info("Checkpoint found, loading state.")
            return json.load(f)
    return {}

@app.command()
def run(
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
    dask_scheduler: str = typer.Option(
        "127.0.0.1:8786",
        "--scheduler",
        help="Address of the Dask scheduler, or 'synchronous' for local testing.",
    ),
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

    if dask_scheduler == "synchronous":
        with dask.config.set(scheduler="synchronous"):
            client = Client()
        logging.info("Using synchronous Dask scheduler for local execution.")
    else:
        client = Client(dask_scheduler)
        logging.info(f"Connected to Dask scheduler at {dask_scheduler}")

    db_manager = DatabaseManager(config.db_path)
    # trainer = PacemakerTrainer(config, MagicMock()) # A real config generator is needed

    state = load_checkpoint()
    start_cycle = state.get("cycle", 1)
    potential_path = state.get("potential_path", "model_initial.yace") # Assume an initial model exists

    max_cycles = 5
    dft_futures = []

    for cycle in range(start_cycle, max_cycles + 1):
        logging.info(f"Starting Active Learning Cycle {cycle}/{max_cycles}")

        # In a real run, training would happen here
        # potential_path = trainer.train(db_manager.get_calculations_for_training())

        quantifier = UncertaintyQuantifier()
        lammps_runner = LammpsRunner(config, potential_path, quantifier)
        simulation_generator = lammps_runner.run()

        for step, (atoms, grade, force_mask) in enumerate(simulation_generator):
            if grade >= config.inference.uncertainty_threshold:
                logging.warning(f"High uncertainty at step {step}, submitting DFT task.")
                future = client.submit(dft_task_wrapper, config, atoms, force_mask)
                dft_futures.append(future)

        # This loop is now only for submitting tasks
        for future in as_completed(dft_futures):
            atoms_result, metadata, mask = future.result()
            db_manager.write_calculation(atoms_result, metadata, mask)
            logging.info(f"DFT result for {metadata.uuid} saved to database.")

        save_checkpoint({"cycle": cycle + 1, "potential_path": potential_path})

    logging.info("Workflow completed.")


if __name__ == "__main__":
    app()
