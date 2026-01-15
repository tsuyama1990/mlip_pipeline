# ruff: noqa: D101, T201
"""Main CLI application for MLIP-AutoPipe."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import typer
import yaml
from ase import Atoms

from mlip_autopipec.config_schemas import (
    DFTConfig,
    DFTExecutable,
    DFTInput,
    InferenceParams,
    MDEnsemble,
    SystemConfig,
    UserConfig,
)
from pydantic import ValidationError

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
        logging.error(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)

    # Instantiate mocked modules
    db_manager = MagicMock()
    dft_runner = MagicMock()
    trainer = MagicMock()

    # The main active learning loop
    max_cycles = 5  # To prevent infinite loops in this mock
    quantifier = UncertaintyQuantifier()
    for cycle in range(1, max_cycles + 1):
        logging.info("-" * 50)
        logging.info(f"Starting Active Learning Cycle {cycle}/{max_cycles}")

        # 1. Train the MLIP using all available data
        logging.info("Step 1: Training the MLIP...")
        trainer.train.return_value = f"model_cycle_{cycle}.yace"
        potential_path = trainer.train()
        logging.info(f"Successfully trained new potential: {potential_path}")

        # 2. Run the MD simulation with the new potential
        logging.info("Step 2: Running MD simulation with uncertainty quantification...")
        lammps_runner = LammpsRunner(
            config=config, potential_path=potential_path, quantifier=quantifier
        )
        simulation_generator = lammps_runner.run()

        simulation_completed = True
        assert config.inference is not None  # For type checker
        for step, (atoms, grade) in enumerate(simulation_generator):
            if grade >= config.inference.uncertainty_threshold:
                logging.warning(
                    f"High uncertainty detected at MD step {step} (grade={grade:.2f})!"
                )
                logging.info("Step 3: Acknowledged high-uncertainty structure.")

                # 3. Run DFT on the high-uncertainty structure
                logging.info("Step 4: Running DFT on the new structure...")
                dft_runner.run.return_value = (atoms, {"energy": -123.45})
                atoms_result, dft_data = dft_runner.run(atoms)
                logging.info("DFT calculation finished.")

                # 4. Save the new data to the database
                logging.info("Step 5: Writing new data to the database...")
                db_manager.write_calculation(atoms=atoms_result, metadata=dft_data)
                logging.info("Database write successful.")

                # Break the inner simulation loop to restart the cycle
                simulation_completed = False
                break
            else:
                # Simulation is stable, continue to the next step
                if step > 0 and step % 100 == 0:
                    logging.info(f"MD Step {step}: Simulation is stable (grade={grade:.2f}).")

        if simulation_completed:
            logging.info("-" * 50)
            logging.info(
                "MD simulation finished without exceeding uncertainty threshold."
            )
            logging.info("Workflow complete.")
            break  # Exit the main loop
    else:
        logging.info("-" * 50)
        logging.info("Reached maximum active learning cycles.")
        logging.info("Workflow finished.")


if __name__ == "__main__":
    app()
