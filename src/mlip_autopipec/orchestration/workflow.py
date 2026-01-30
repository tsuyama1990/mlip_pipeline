import logging

from mlip_autopipec.domain_models.config import Config, MDParams
from mlip_autopipec.domain_models.job import JobResult, JobStatus
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

logger = logging.getLogger(__name__)


def run_one_shot(config: Config) -> JobResult:
    """
    Execute the One-Shot Pipeline: Build -> Run -> Report.
    """
    logger.info("Starting One-Shot Pipeline (Cycle 02)...")

    # 1. Generate Structure
    # Use first element in potential config or default to Si
    element = config.potential.elements[0] if config.potential.elements else "Si"
    logger.info(f"Building bulk structure for {element}...")

    builder = StructureBuilder()
    # Hardcoded parameters for now as per "One-Shot" simplicity, or could come from Config if we added StructureGenConfig
    # Using defaults for Si
    structure = builder.build_bulk(element, "diamond", 5.43)

    # Rattle
    structure = builder.apply_rattle(structure, stdev=0.1, seed=config.potential.seed)

    # 2. Run MD
    logger.info("Running MD simulation...")
    runner = LammpsRunner(config.lammps)

    # Use defaults or allow Config to override if we added fields to Config
    # Creating params
    params = MDParams(
        temperature=300.0,
        n_steps=1000,
        timestep=0.001
    )

    result = runner.run(structure, params)

    # 3. Report
    logger.info(f"Job {result.job_id} finished with status {result.status.value}")
    if result.status == JobStatus.COMPLETED:
        logger.info(f"Final Structure: {result.final_structure.get_chemical_formula() if hasattr(result, 'final_structure') and result.final_structure else 'N/A'}")
    else:
        logger.error(f"Job Failed: {result.log_content}")

    return result
