import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.training.pacemaker import PacemakerWrapper
from mlip_autopipec.utils.embedding import EmbeddingExtractor

logger = logging.getLogger(__name__)


class SelectionStrategy(ABC):
    """
    Abstract base class for selection strategies in active learning.
    """

    @abstractmethod
    def select(self, candidates: list[Atoms], potential_path: Path) -> list[Atoms]:
        """
        Selects a subset of candidates for DFT calculation.

        Args:
            candidates: List of candidate structures (usually extracted from MD).
            potential_path: Path to the current potential (needed for active set selection).

        Returns:
            List of selected Atoms objects.
        """


class GammaSelectionStrategy(SelectionStrategy):
    """
    Selection strategy based on extrapolation grade (Gamma) and Pacemaker's active set selection.
    """

    def __init__(
        self, pacemaker_wrapper: PacemakerWrapper, embedding_config: EmbeddingConfig
    ) -> None:
        self.pacemaker = pacemaker_wrapper
        self.embedding_config = embedding_config
        self.extractor = EmbeddingExtractor(embedding_config)

    def select(self, candidates: list[Atoms], potential_path: Path) -> list[Atoms]:
        """
        Selects structures using active set selection.
        """
        if not candidates:
            return []

        # Write candidates to temporary file
        candidates_path = self.pacemaker.work_dir / "candidates_temp.xyz"
        # type: ignore[no-untyped-call]
        write(str(candidates_path), candidates, format="extxyz")

        # 1. Active Set Selection (Downsampling)
        try:
            indices = self.pacemaker.select_active_set(candidates_path, potential_path)
            selected_atoms = [candidates[i] for i in indices]
            logger.info(f"Active Set Selection: Reduced {len(candidates)} -> {len(selected_atoms)}")
        except Exception:
            logger.exception("Active set selection failed. Falling back to all candidates.")
            return candidates
        else:
            return selected_atoms
