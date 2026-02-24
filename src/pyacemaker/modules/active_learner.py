"""Active Learner module implementation."""

import heapq
from collections.abc import Iterable
from pathlib import Path

from pyacemaker.core.config import OrchestratorConfig
from pyacemaker.core.interfaces import Oracle
from pyacemaker.core.utils import atoms_to_metadata
from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.oracle.dataset import DatasetManager


class ActiveLearner:
    """Active Learner for selecting informative structures."""

    def __init__(self, config: OrchestratorConfig, oracle: Oracle) -> None:
        """Initialize Active Learner."""
        self.config = config
        self.oracle = oracle
        self.dataset_manager = DatasetManager()

    def run_loop(self, pool_path: Path, work_dir: Path) -> tuple[Path, Path]:
        """Run the active learning loop.

        Reads from pool, labels with Oracle, saves to DFT dataset.
        """
        final_model_path = work_dir / "final_model.model"
        dft_dataset_path = work_dir / "dft_dataset.xyz"

        # Ensure dummy model exists for flow
        if not final_model_path.exists():
            final_model_path.touch()

        # Process pool -> Label -> Save
        # Load pool as stream
        pool_stream = self.dataset_manager.load_iter(pool_path)
        metadata_stream = (atoms_to_metadata(a) for a in pool_stream)

        # Label with Oracle
        labeled_stream = self.oracle.compute_batch(metadata_stream)

        # Save to DFT dataset
        # Note: We use .xyz extension in variable name but save as pickle format via manager
        self.dataset_manager.save_metadata_stream(labeled_stream, dft_dataset_path)

        return final_model_path, dft_dataset_path

    def select_batch(
        self,
        candidates: Iterable[StructureMetadata],
        n_select: int,
        threshold: float | None = None,
    ) -> list[StructureMetadata]:
        """Select the most informative structures from candidates using uncertainty.

        Uses heapq.nlargest for O(K) memory efficiency where K=n_select.
        Supports filtering by uncertainty threshold (strictly greater than).

        Args:
            candidates: Iterable of candidate structures (streaming).
            n_select: Number of structures to select.
            threshold: Optional uncertainty threshold. Only structures > threshold are considered.

        Returns:
            List of selected structures sorted by uncertainty (descending).

        """
        if n_select <= 0:
            return []

        # Filter candidates lazily if threshold is provided
        filtered_candidates = candidates
        if threshold is not None:
            filtered_candidates = (
                c
                for c in candidates
                if c.uncertainty_state
                and c.uncertainty_state.gamma_max is not None
                and c.uncertainty_state.gamma_max > threshold
            )

        def get_uncertainty(s: StructureMetadata) -> float:
            """Extract uncertainty metric for sorting."""
            if s.uncertainty_state and s.uncertainty_state.gamma_max is not None:
                return s.uncertainty_state.gamma_max
            return -1.0

        # Efficient O(N) time, O(K) memory selection
        return heapq.nlargest(n_select, filtered_candidates, key=get_uncertainty)
