"""Structure Generator module."""

import logging

from mlip_autopipec.domain_models.config import ExplorationConfig
from mlip_autopipec.domain_models.structure import Candidate, CandidateStatus, Structure
from mlip_autopipec.modules.structure_gen.strategies import ColdStartStrategy, RandomPerturbationStrategy

logger = logging.getLogger(__name__)


class StructureGenerator:
    """Facade for structure generation strategies."""

    def __init__(self, config: ExplorationConfig) -> None:
        self.config = config
        self.cold_start = ColdStartStrategy()
        self.perturbation = RandomPerturbationStrategy()

    def generate_initial_set(self) -> list[Candidate]:
        """Generate initial set of candidates."""
        logger.info(f"Generating initial structures for {self.config.composition}")
        structures = self.cold_start.generate(self.config)

        candidates = []
        for s in structures:
            candidates.append(
                Candidate(
                    **s.model_dump(),
                    source="cold_start",
                    status=CandidateStatus.PENDING,
                    priority=1.0  # High priority for initial set
                )
            )
        logger.info(f"Generated {len(candidates)} candidates.")
        return candidates

    def apply_strategy(self, structures: list[Structure], strategy_name: str = "random") -> list[Candidate]:
        """Apply a strategy to a list of structures."""
        candidates = []
        # Currently we only have random perturbation logic

        for s in structures:
            if strategy_name == "random":
                new_s = self.perturbation.apply(s, self.config)
                candidates.append(
                    Candidate(
                        **new_s.model_dump(),
                        source="random_perturbation",
                        status=CandidateStatus.PENDING,
                        priority=0.5
                    )
                )
        return candidates
