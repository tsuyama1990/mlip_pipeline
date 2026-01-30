import logging
import uuid
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder

logger = logging.getLogger(__name__)


def run_one_shot(config: Config) -> JobResult:
    """
    Execute the One-Shot Pipeline:
    Generate Structure -> Run MD -> Return Result.
    """
    logger.info("Starting One-Shot Pipeline")

    # 1. Generate Structure
    element = config.structure_gen.composition or "Si"
    lattice_const = config.structure_gen.lattice_constant

    logger.info(f"Building initial structure for {element} (a={lattice_const})")

    builder = StructureBuilder(seed=config.potential.seed)

    structure = builder.build_bulk(element, "diamond", lattice_const)

    if config.structure_gen.rattle_amplitude > 0:
        structure = builder.apply_rattle(structure, config.structure_gen.rattle_amplitude)

    # 2. Setup Runner
    runner = LammpsRunner(config.lammps)

    # 3. Define Job
    job_id = str(uuid.uuid4())[:8]
    work_dir = Path("_work_md") / f"job_{job_id}"

    # Use config parameters instead of hardcoded ones
    params = config.structure_gen.md_params

    # 4. Execute
    logger.info(f"Launching MD Job {job_id}...")
    result = runner.run(structure, params, work_dir)

    logger.info(f"Job {job_id} finished with status {result.status}")
    return result
