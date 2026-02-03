import logging
from pathlib import Path

from mlip_autopipec.config.config_model import SelectionConfig
from mlip_autopipec.domain_models.structures import CandidateStructure
from mlip_autopipec.orchestration.interfaces import Selector

logger = logging.getLogger(__name__)


class ActiveSetSelector(Selector):
    def __init__(self, config: SelectionConfig) -> None:
        self.config = config

    def select(
        self,
        candidates: list[CandidateStructure],
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        """
        Selects candidates based on uncertainty or simply top N.
        """
        if not candidates:
            return []

        # Sort by uncertainty descending if available
        # We assume higher uncertainty = higher priority for labeling
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.metadata.uncertainty if c.metadata.uncertainty is not None else -1.0,
            reverse=True,
        )

        # Cutoff
        limit = self.config.max_structures
        selected = sorted_candidates[:limit]

        logger.info(
            f"Selected {len(selected)} out of {len(candidates)} candidates "
            f"(Method: {self.config.method}, Max: {limit})"
        )

        return selected
