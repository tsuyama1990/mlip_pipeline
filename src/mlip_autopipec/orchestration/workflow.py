import logging
import uuid
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.material import get_material_properties
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner, MDParams
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder

logger = logging.getLogger("mlip_autopipec.orchestration")


def run_one_shot(config: Config) -> None:
    """
    Run the Cycle 02 One-Shot pipeline.
    """
    logger.info("Starting One-Shot Pipeline (Cycle 02)")

    # 1. Generate Structure
    builder = StructureBuilder()
    if not config.potential.elements:
        raise ValueError("No elements defined in config")

    element = config.potential.elements[0]  # Assuming first element
    logger.info(f"Generating bulk structure for {element}")

    mat_props = get_material_properties(element)
    if not mat_props:
        logger.warning(f"Material {element} not found in database. Using defaults.")
        lattice_constant = 5.43
        crystal = "diamond"
    else:
        lattice_constant = mat_props.lattice_constant
        crystal = mat_props.crystal_structure

    structure = builder.build_bulk(element, crystal, lattice_constant, cubic=True)

    # Apply rattle
    structure = builder.apply_rattle(structure, stdev=0.01)

    # 2. Setup LAMMPS
    runner = LammpsRunner(config.lammps)

    # Define simple params
    params = MDParams(
        temperature=300,
        n_steps=5000,
        timestep=0.001
    )

    # Create unique job dir
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    work_dir = Path("_work_md") / job_id

    logger.info(f"Running MD simulation in {work_dir}")

    # 3. Run
    result = runner.run(structure, work_dir, params)

    if result.status == JobStatus.COMPLETED:
        logger.info("Simulation Completed: Status DONE")
        if result.final_structure:
             logger.info(f"Final Structure has {len(result.final_structure.positions)} atoms")
    else:
        logger.error(f"Simulation Failed: {result.status}")
        logger.error(f"Log Tail: \n{result.log_content}")
        raise RuntimeError(f"Job failed with status {result.status}")
