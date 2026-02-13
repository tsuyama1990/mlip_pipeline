import logging
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models.config import ActiveLearningConfig, TrainerConfig
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.domain_models.enums import ActiveSetMethod

logger = logging.getLogger(__name__)


class ActiveSelector:
    """
    Selects the most informative candidates from a local pool using D-Optimality or Random selection.
    """

    def __init__(self, config: ActiveLearningConfig, trainer_config: TrainerConfig) -> None:
        self.config = config
        self.trainer_config = trainer_config

    def select_batch(self, candidates: Iterable[Structure]) -> Iterator[Structure]:
        """
        Selects a subset of candidates based on the configured active set method.

        Args:
            candidates: An iterable of candidate structures.

        Returns:
            An iterator of selected structures.
        """
        candidates_list = list(candidates)
        if not candidates_list:
            logger.warning("No candidates provided to ActiveSelector.")
            return

        method = self.trainer_config.active_set_method
        count = self.trainer_config.n_active_set_per_halt

        # If count > available, return all
        if count >= len(candidates_list):
            logger.info(f"ActiveSelector: Selecting all {len(candidates_list)} candidates (requested {count}).")
            for c in candidates_list:
                c.provenance += "_active_all"
                yield c
            return

        if method == ActiveSetMethod.RANDOM:
            yield from self._select_random(candidates_list, count)
        elif method == ActiveSetMethod.MAXVOL:
             # TODO: Implement proper MaxVol wrapper calling pacemaker
             # For now, fallback to random or mock MaxVol
            logger.info("ActiveSelector: MAXVOL requested, falling back to Random/Mock for Cycle 06 implementation.")
            yield from self._select_random(candidates_list, count)
        else:
             # Default: Take first N (or random)
            yield from self._select_random(candidates_list, count)

    def _select_random(self, candidates: list[Structure], count: int) -> Iterator[Structure]:
        import random
        selected = random.sample(candidates, count)
        logger.info(f"ActiveSelector: Randomly selected {count} structures.")
        for s in selected:
            s.provenance += "_active_random"
            yield s
