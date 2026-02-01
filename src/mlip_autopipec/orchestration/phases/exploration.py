import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.domain_models.dynamics import LammpsResult
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

logger = logging.getLogger("mlip_autopipec.phases.exploration")

class ExplorationPhase:
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> LammpsResult:
        """
        Execute the Exploration Phase:
        1. Generate a starting structure.
        2. Run Molecular Dynamics (MD) with the current potential.
        """
        logger.info("Running Exploration Phase...")

        # 1. Generate Structure
        gen_config = config.structure_gen
        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)
        logger.info(f"Generated structure: {structure.get_chemical_formula()}")

        # 2. Run MD
        md_config = config.md
        # Inject uncertainty threshold from Orchestrator config
        md_config.uncertainty_threshold = config.orchestrator.uncertainty_threshold

        # Setup Runner
        # We pass work_dir as base_work_dir. The runner will create subdirectories (e.g. job_timestamp)
        # But we usually want it in 'md_run'.
        md_work_dir = work_dir / "md_run"
        md_work_dir.mkdir(exist_ok=True)

        runner = LammpsRunner(
            config=config.lammps,
            potential_config=config.potential,
            base_work_dir=md_work_dir
        )

        result = runner.run(structure, md_config, potential_path=state.latest_potential_path)

        return result
