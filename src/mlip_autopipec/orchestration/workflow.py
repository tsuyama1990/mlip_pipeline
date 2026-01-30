import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import LammpsResult
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

logger = logging.getLogger(__name__)


def run_one_shot(config: Config) -> LammpsResult:
    """
    Execute the One-Shot Pipeline: Generate -> Run -> Parse.
    """
    logger.info("Starting One-Shot Pipeline (Cycle 02)")

    # 1. Structure Generation
    builder = StructureBuilder()

    # Defaults or from config
    element = config.exploration.composition
    crystal_structure = "diamond"
    lattice_constant = config.exploration.lattice_constant

    logger.info(f"Building initial structure: {element} ({crystal_structure})")
    structure = builder.build_bulk(element, crystal_structure, lattice_constant)

    if config.exploration.rattle_amplitude > 0:
        logger.info(f"Applying rattle with amplitude {config.exploration.rattle_amplitude}")
        structure = builder.apply_rattle(structure, config.exploration.rattle_amplitude)

    # 2. Run MD
    runner = LammpsRunner(config.lammps)

    md_params = config.exploration.md_params

    logger.info("Running MD Simulation...")
    work_dir = Path("_work_md/job_one_shot")

    result = runner.run(structure, md_params, work_dir=work_dir)

    logger.info(f"Simulation Completed: Status {result.status.value}")
    if result.status.value == "FAILED":
        logger.error(f"Reason: {result.log_content}")

    return result
