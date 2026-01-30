import logging
import typer
from mlip_autopipec.domain_models.config import Config, MDParams
from mlip_autopipec.domain_models.job import JobResult, JobStatus
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

logger = logging.getLogger("mlip_autopipec")

def run_one_shot(config: Config) -> JobResult:
    """
    Execute the One-Shot Pipeline (Cycle 02).
    1. Generate Structure (Bulk Si)
    2. Run MD (LAMMPS)
    3. Output Result
    """
    logger.info("Starting One-Shot Pipeline (Cycle 02)")

    # 1. Structure Generation
    logger.info("Generating initial structure (Bulk Si)...")
    builder = StructureBuilder()
    structure = builder.build_bulk("Si", "diamond", 5.43)

    # Apply some thermal noise to make it interesting
    structure = builder.apply_rattle(structure, stdev=0.1)

    # 2. Setup MD Parameters
    # For Cycle 02, we use fixed parameters for the test
    md_params = MDParams(
        temperature=300.0,
        n_steps=1000,
        timestep=0.001,
        ensemble="NVT"
    )

    # 3. Execution
    logger.info("Initializing LammpsRunner...")
    runner = LammpsRunner(config.lammps)

    logger.info(f"Running MD simulation (Steps={md_params.n_steps}, Temp={md_params.temperature}K)...")
    result = runner.run(structure, md_params)

    # 4. Output
    if result.status == JobStatus.COMPLETED:
        logger.info(f"Simulation Completed: Status {result.status.value}")
        typer.secho(f"Simulation Completed: Status {result.status.value}", fg=typer.colors.GREEN)
        logger.info(f"Trajectory saved to: {result.trajectory_path}")
        logger.info(f"Duration: {result.duration_seconds:.2f}s")
    else:
        logger.error(f"Simulation Failed: Status {result.status.value}")
        typer.secho(f"Simulation Failed: Status {result.status.value}", fg=typer.colors.RED)
        logger.error(f"Log Tail:\n{result.log_content}")
        typer.echo(f"Log Tail:\n{result.log_content}")

    return result
