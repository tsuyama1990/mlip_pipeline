import logging
import random
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models.config import ActiveLearningConfig, TrainerConfig
from mlip_autopipec.domain_models.enums import ActiveSetMethod
from mlip_autopipec.domain_models.structure import Structure

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
        method = self.trainer_config.active_set_method
        count = self.trainer_config.n_active_set_per_halt

        if method == ActiveSetMethod.RANDOM:
            yield from self._select_random(candidates, count)
        elif method == ActiveSetMethod.MAXVOL:
            # MAXVOL implementation requires full batch loading or complex incremental update.
            # Currently disabled as per strict memory safety requirement until efficient implementation.
            msg = "ActiveSetMethod.MAXVOL is not yet implemented. Please use ActiveSetMethod.RANDOM."
            logger.error(msg)
            raise NotImplementedError(msg)
        else:
            # Default: Take first N
            yield from self._select_first_n(candidates, count)

    def _select_random(self, candidates: Iterable[Structure], count: int) -> Iterator[Structure]:
        """
        Selects random candidates using reservoir sampling to support streaming.
        """
        reservoir: list[Structure] = []

        for i, candidate in enumerate(candidates):
            if len(reservoir) < count:
                reservoir.append(candidate)
            else:
                # Replace elements with decreasing probability
                j = random.randint(0, i) # noqa: S311
                if j < count:
                    reservoir[j] = candidate

        logger.info(f"ActiveSelector: Randomly selected {len(reservoir)} structures.")
        for s in reservoir:
            s.provenance += "_active_random"
            yield s

    def _select_first_n(self, candidates: Iterable[Structure], count: int) -> Iterator[Structure]:
        """
        Selects the first N candidates from the stream.
        """
        selected_count = 0
        for s in candidates:
            if selected_count >= count:
                break
            s.provenance += "_active_first"
            yield s
            selected_count += 1

        logger.info(f"ActiveSelector: Selected first {selected_count} structures.")
