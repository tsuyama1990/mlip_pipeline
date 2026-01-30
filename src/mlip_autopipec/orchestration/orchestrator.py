import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory

logger = logging.getLogger("mlip_autopipec")


class Orchestrator:
    """
    Orchestrates the active learning pipeline (or parts of it).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def run_pipeline(self) -> JobResult:
        """
        Execute the pipeline configured in Config.
        Currently supports the One-Shot cycle: Generate -> Dynamics.
        """
        logger.info("Starting Pipeline Execution")

        # 1. Structure Generation
        logger.info("Phase 1: Structure Generation")
        gen_config = self.config.structure_gen
        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)
        logger.info(f"Generated structure: {structure.get_chemical_formula()}")

        # 2. Dynamics (MD)
        logger.info("Phase 2: Molecular Dynamics")
        md_config = self.config.md
        runner = LammpsRunner(self.config.lammps, self.config.potential)

        # md_config is MDConfig, which is aliased to MDParams in config.py
        # LammpsRunner expects MDParams
        result = runner.run(structure, md_config)

        logger.info(f"Pipeline Finished with status: {result.status.value}")
        return result
