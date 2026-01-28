import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig

# from mlip_autopipec.utils.embedding import EmbeddingExtractor # unused

logger = logging.getLogger(__name__)


class CandidateProcessor:
    """
    Processes raw MD output (uncertain structures) into training candidates.
    Applies:
    1. Clustering/Extraction (Cluster-in-box)
    2. Force Masking (if enabled)
    3. Filtering (e.g. min distance)
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.embedding_config = EmbeddingConfig()

    def extract_candidates(
        self, uncertain_structures: list[Path], reference_atoms: Atoms
    ) -> list[tuple[Atoms, dict]]:
        """
        Extracts candidate structures from dump files.

        Args:
            uncertain_structures: List of paths to LAMMPS dump files.
            reference_atoms: The structure used to start the MD (provides cell/pbc info).

        Returns:
            List of (Atoms, metadata) tuples.
        """
        candidates = []
        # extractor = EmbeddingExtractor(self.embedding_config) # unused

        for dump_path in uncertain_structures:
            try:
                # Read all frames from dump
                # ase.io.read returns list if index=':'
                # type: ignore[no-untyped-call]
                frames = read(dump_path, index=":")

                # Handle single Atoms return (though index=":" usually returns list)
                if isinstance(frames, Atoms):
                    frames = [frames]

                if not frames:
                    continue

                # For each frame (uncertainty event)
                for i, atoms in enumerate(frames):
                    # In active learning loop, we often extract the local environment
                    # around the atom with high uncertainty.
                    # However, typical AL with potentials like MACE/NequIP might just take the whole frame
                    # or a large cluster.
                    # Here we assume we take the whole frame for simplicity, unless clustering logic is added.

                    # Ensure PBC and Cell are correct (sometimes lost in dumps depending on format)
                    if np.allclose(atoms.cell.lengths(), 0):
                        atoms.set_cell(reference_atoms.get_cell())
                        atoms.set_pbc(reference_atoms.get_pbc())

                    # Optional: Extract cluster around high-uncertainty atom if we knew which one it was.
                    # The dump file might contain 'c_gamma' column.
                    if "c_gamma" in atoms.arrays:
                        gammas = atoms.arrays["c_gamma"]
                        max_idx = int(np.argmax(gammas))
                        # cluster = extractor.extract(atoms, max_idx) # Implement cluster extraction if needed
                        # candidates.append((cluster, {"origin": dump_path.name, "frame": i, "gamma": gammas[max_idx]}))
                        candidates.append((atoms, {"origin": dump_path.name, "frame": i, "gamma": float(gammas[max_idx])}))
                    else:
                        candidates.append((atoms, {"origin": dump_path.name, "frame": i}))

            except Exception:
                logger.exception(f"Failed to process dump file: {dump_path}")

        return candidates
