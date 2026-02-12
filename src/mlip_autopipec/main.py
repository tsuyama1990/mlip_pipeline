from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.core.config_parser import load_config
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.datastructures import WorkflowState
from mlip_autopipec.domain_models.enums import TaskStatus

app = typer.Typer()


@app.command()
def init(
    output: Path = typer.Option("config.yaml", help="Output config file path"),  # noqa: B008
) -> None:
    """
    Generate a default configuration file.
    """
    default_config = {
        "orchestrator": {"work_dir": "mlip_run", "max_iterations": 10},
        "generator": {"type": "RANDOM", "num_structures": 10},
        "oracle": {"type": "QUANTUM_ESPRESSO", "command": "pw.x", "mixing_beta": 0.7},
        "trainer": {"type": "PACEMAKER", "r_cut": 5.0, "max_deg": 3},
    }

    with output.open("w") as f:
        yaml.dump(default_config, f, sort_keys=False)
    typer.echo(f"Created default configuration at {output}")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to config.yaml"),  # noqa: B008
) -> None:
    """
    Run the MLIP pipeline.
    """
    try:
        config = load_config(config_path)

        # Ensure work_dir exists
        if not config.orchestrator.work_dir.exists():
            config.orchestrator.work_dir.mkdir(parents=True)

        setup_logging(config.orchestrator.work_dir)

        # State Manager
        sm = StateManager(config.orchestrator.work_dir)
        state = sm.load_state()

        if state:
            typer.echo(f"Resuming from iteration {state.iteration}")
        else:
            typer.echo("Starting new run")
            state = WorkflowState(iteration=0, status=TaskStatus.RUNNING)
            sm.save_state(state)

        typer.echo("Configuration loaded successfully.")
        typer.echo(f"Work directory: {config.orchestrator.work_dir}")

    except ValidationError as e:
        typer.echo(f"Configuration Validation Error:\n{e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"Runtime Error: {e}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
