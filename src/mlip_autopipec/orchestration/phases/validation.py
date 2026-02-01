import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config, BulkStructureGenConfig
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.validation.runner import ValidationRunner

logger = logging.getLogger("mlip_autopipec.phases.validation")

class ValidationPhase:
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> bool:
        logger.info("Validating potential...")

        if not state.latest_potential_path:
            return False

        # Generate bulk structure
        gen_config = config.structure_gen
        if isinstance(gen_config, BulkStructureGenConfig):
            gen_config = gen_config.model_copy(update={"rattle_stdev": 0.0})

        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)

        runner = ValidationRunner(
            val_config=config.validation,
            pot_config=config.potential,
            potential_path=state.latest_potential_path
        )

        result = runner.validate(structure)
        logger.info(f"Validation Result: {result.overall_status}")

        return True
