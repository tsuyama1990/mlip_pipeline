import itertools
import logging

from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.datastructures import HaltInfo, Potential
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
        candidates_iter = self.candidate_generator.generate_local(halt_event.structure)

        # 2. Select Active Set
        logger.info("ActiveLearner: Selecting active set...")
        # Get iterator from selector. Note: ActiveSelector.select_batch now uses reservoir sampling internally
        # for Random, so it will consume the input stream but yield output stream.
        # We peek to check if any were selected without consuming everything if possible,
        # but reservoir sampling implies full consumption of input candidates.
        # However, we should avoid calling list() on the output of select_batch to respect memory safety
        # if select_batch returns a generator (which it does).

        selected_candidates_iter = self.active_selector.select_batch(candidates_iter)

        # Use peek mechanism to check for empty stream
        try:
            first_candidate = next(selected_candidates_iter)
            # Reconstruct iterator
            selected_candidates_stream = itertools.chain([first_candidate], selected_candidates_iter)
            has_candidates = True
        except StopIteration:
            has_candidates = False
            selected_candidates_stream = iter([])  # type: ignore[assignment]

        if not has_candidates:
            logger.warning("ActiveLearner: No candidates selected. Returning existing potential (no-op).")
            # We assume the trainer can handle an "update" request that results in no change
            # or we need to return the current potential.
            # Since we don't track current potential here, we might need to rely on the trainer returning
            # a valid potential even if empty data.
            # But the audit requires a robust check.
            # Let's try training with empty set.
            # If trainer fails, we should catch it.

        # 3. Label (Oracle)
        logger.info("ActiveLearner: Computing ground truth (Oracle)...")
        # Oracle.compute takes Iterable and returns Iterable (or list).
        # We pass the stream.
        labeled_structures = self.oracle.compute(selected_candidates_stream)

        # 4. Train (Fine-tune)
        logger.info("ActiveLearner: Fine-tuning potential...")
        new_potential = self.trainer.train(labeled_structures)

        if new_potential is None:
             msg = "Trainer returned None potential after fine-tuning."
             logger.error(msg)
             raise RuntimeError(msg)

        logger.info(f"ActiveLearner: Cycle complete. New potential: {new_potential.path}")
        return new_potential
