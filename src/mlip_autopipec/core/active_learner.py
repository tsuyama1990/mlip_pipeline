import logging

from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.datastructures import HaltInfo, Potential, Structure
from mlip_autopipec.generator.candidate_generator import CandidateGenerator
from mlip_autopipec.generator.interface import BaseGenerator
from mlip_autopipec.oracle.interface import BaseOracle
from mlip_autopipec.trainer.active_selector import ActiveSelector
from mlip_autopipec.trainer.interface import BaseTrainer

logger = logging.getLogger(__name__)

class ActiveLearner:
    """
    Orchestrates the Local Learning Loop (Halt & Diagnose).
    """

    def __init__(
        self,
        config: GlobalConfig,
        generator: BaseGenerator,
        oracle: BaseOracle,
        trainer: BaseTrainer,
        candidate_generator: CandidateGenerator | None = None,
        active_selector: ActiveSelector | None = None,
    ) -> None:
        self.config = config
        self.oracle = oracle
        self.trainer = trainer
        # generator argument kept for potential future use or policy access, though we use specialized CandidateGenerator
        self.generator = generator

        # Specialized components for active learning
        self.candidate_generator = candidate_generator or CandidateGenerator(
            config.active_learning
        )
        self.active_selector = active_selector or ActiveSelector(
            config.active_learning, config.trainer
        )

    def process_halt(self, halt_event: HaltInfo) -> Potential:
        """
        Handles a simulation halt by generating local candidates, labeling them,
        and fine-tuning the potential.

        Args:
            halt_event: The event information containing the halt structure and reason.

        Returns:
            The updated Potential object.
        """
        logger.info(f"ActiveLearner: Processing halt at step {halt_event.step} (gamma={halt_event.max_gamma:.2f})")

        # 1. Generate Candidates
        logger.info("ActiveLearner: Generating local candidates...")
        # Add explicit type hint for candidates_iter for better readability and IDE support
        candidates_iter = self.candidate_generator.generate_local(halt_event.structure)

        # 2. Select Active Set
        logger.info("ActiveLearner: Selecting active set...")
        selected_candidates: list[Structure] = list(self.active_selector.select_batch(candidates_iter))
        logger.info(f"ActiveLearner: Selected {len(selected_candidates)} candidates for labeling.")

        if not selected_candidates:
            logger.warning("ActiveLearner: No candidates selected. Returning existing potential (no-op).")
            # See previous note about returning potential.
            # We don't have the "current" potential explicitly passed here, so we might need to rely on
            # what the trainer returns for an empty training set, or what we can infer.
            # However, to be safe and type correct, we proceed to compute step with empty list
            # and let the Trainer handle "no data".

        # 3. Label (Oracle)
        logger.info("ActiveLearner: Computing ground truth (Oracle)...")
        labeled_structures = self.oracle.compute(selected_candidates)

        # 4. Train (Fine-tune)
        logger.info("ActiveLearner: Fine-tuning potential...")
        # Trainer.train should handle fine-tuning
        new_potential = self.trainer.train(labeled_structures)

        logger.info(f"ActiveLearner: Cycle complete. New potential: {new_potential.path}")
        return new_potential
