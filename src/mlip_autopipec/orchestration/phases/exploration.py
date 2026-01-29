"""Exploration Phase implementation."""

import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.modules.structure_gen.generator import StructureGenerator

logger = logging.getLogger(__name__)


class ExplorationPhase:
    """Handles the exploration phase of the active learning loop."""

    name = WorkflowPhase.EXPLORATION

    def execute(self, state: WorkflowState, config: Config) -> None:
        """Execute the exploration phase logic."""
        logger.info("Executing Exploration Phase")

        # Initialize generator
        generator = StructureGenerator(config.structure_gen)

        # Logic:
        # If candidates are empty, we generate them.
        if not state.candidates:
            if config.structure_gen.composition:
                # Cold Start
                logger.info("No candidates found. Performing Cold Start.")
                candidates = generator.generate_initial_set()
                state.candidates = candidates
            else:
                # Future: Pick from dataset and perturb
                logger.warning("No composition for Cold Start. Skipping generation.")

        # In a real cycle, we would transition to ORACLE or SELECTION.
        # For Cycle 02, we stay here or let WorkflowManager handle transition.
        # We assume the goal of this phase is just to populate candidates.
        logger.info(f"Exploration Phase complete. Candidates: {len(state.candidates)}")
