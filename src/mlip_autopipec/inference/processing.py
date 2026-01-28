import logging
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from mlip_autopipec.config.schemas.common import EmbeddingConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.utils.embedding import EmbeddingExtractor

logger = logging.getLogger(__name__)


class CandidateProcessor:
    """
    Service for processing and extracting candidate structures from simulation dumps.
    Encapsulates logic for handling different engine outputs (LAMMPS vs EON).
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.embedding_config = EmbeddingConfig()
        self.extractor = EmbeddingExtractor(self.embedding_config)

    def extract_candidates(
        self, dump_paths: list[Path], start_atoms: Atoms
    ) -> list[tuple[Atoms, dict[str, Any]]]:
        """
        Extracts candidates from a list of dump files.

        Args:
            dump_paths: List of paths to dump files.
            start_atoms: The initial atoms object (used for species mapping in LAMMPS).

        Returns:
            List of tuples (Atoms, metadata_dict) ready for database insertion.
        """
        candidates: list[tuple[Atoms, dict[str, Any]]] = []

        for dump_path in dump_paths:
            try:
                frame: Atoms | None = None

                # Differentiate extraction based on engine
                if self.config.active_engine == "eon":
                    # EON produces single-frame .con files usually.
                    frame = read(dump_path)
                else:
                    # LAMMPS: Read ONLY the last frame (index=-1) to avoid OOM
                    frame = read(dump_path, index=-1, format="lammps-dump-text")

                if frame is None:
                    continue

                # Simplified: Re-assign symbols based on types if available (LAMMPS specific)
                if self.config.active_engine == "lammps":
                    species = sorted(set(start_atoms.get_chemical_symbols()))
                    if "type" in frame.arrays:
                        types = frame.arrays["type"]
                        symbols = [species[t - 1] for t in types]
                        frame.set_chemical_symbols(symbols)

                # Logic for extraction
                extracted_atoms: Atoms
                if "c_gamma" in frame.arrays:
                    gammas = frame.arrays["c_gamma"]
                    max_idx = int(gammas.argmax())
                    extracted_atoms = self.extractor.extract(frame, max_idx)
                else:
                    # For EON, or if missing gamma, take the whole frame
                    extracted_atoms = frame

                # Metadata template (caller should enrich if needed, but we provide base status)
                metadata: dict[str, Any] = {"status": "screening", "config_type": "active_learning"}
                candidates.append((extracted_atoms, metadata))

            except Exception:
                logger.exception(f"Failed to process dump file {dump_path}")

        return candidates
