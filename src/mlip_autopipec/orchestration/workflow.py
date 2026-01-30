import logging
import uuid
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner, MDParams
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder

logger = logging.getLogger(__name__)


def run_one_shot(config: Config) -> JobResult:
    """
    Execute the One-Shot Pipeline:
    Generate Structure -> Run MD -> Return Result.
    """
    logger.info("Starting One-Shot Pipeline")

    # 1. Generate Structure
    # Use config.structure_gen.composition (Cycle 1 legacy name 'structure_gen' in my Config update)
    # The config field is 'structure_gen', type ExplorationConfig.

    element = config.structure_gen.composition or "Si"
    logger.info(f"Building initial structure for {element}")

    builder = StructureBuilder(seed=config.potential.seed)

    # Defaults for demo
    structure = builder.build_bulk(element, "diamond", 5.43)

    if config.structure_gen.rattle_amplitude > 0:
        structure = builder.apply_rattle(structure, config.structure_gen.rattle_amplitude)

    # 2. Setup Runner
    runner = LammpsRunner(config.lammps)

    # 3. Define Job
    job_id = str(uuid.uuid4())[:8]
    work_dir = Path("_work_md") / f"job_{job_id}"

    params = MDParams(
        temperature=300,
        n_steps=1000,
        timestep=0.001
    )

    # 4. Execute
    logger.info(f"Launching MD Job {job_id}...")
    result = runner.run(structure, params, work_dir)

    logger.info(f"Job {job_id} finished with status {result.status}")
    return result
