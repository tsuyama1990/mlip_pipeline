from typing import Any, Iterable
import logging
from ase import Atoms
from mlip_autopipec.orchestration.database import DatabaseManager

logger = logging.getLogger(__name__)

class CandidateManager:
    """
    Manages candidate data persistence logic.
    Separates business logic (defaults, validation) from raw DB access.
    """
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def _prepare_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        meta = metadata.copy()
        meta.setdefault("status", "pending")
        meta.setdefault("generation", 0)
        meta.setdefault("config_type", "candidate")
        return meta

    def create_candidate(self, atoms: Atoms, metadata: dict[str, Any] | None = None) -> None:
        """
        Creates a single candidate structure in the database.
        """
        try:
            meta = self._prepare_metadata(metadata)
            self.db.add_structure(atoms, meta)
            logger.debug("Created candidate structure.")
        except Exception as e:
            logger.error(f"Failed to create candidate: {e}")
            raise

    def create_candidates(self, candidates: Iterable[tuple[Atoms, dict[str, Any] | None]]) -> None:
        """
        Creates multiple candidate structures in a batch.
        Accepts an iterable to support streaming and avoid OOM.
        """
        try:
            # We process the iterable and prepare metadata
            # For efficiency, we can delegate batch saving to database if it supports it.
            # database.save_candidates expects (Atoms, dict).

            prepared_batch = (
                (atoms, self._prepare_metadata(meta))
                for atoms, meta in candidates
            )

            # Note: save_candidates in DatabaseManager currently iterates.
            # If the iterable is huge, we might want to chunk it here or in DatabaseManager.
            # For now, we pass the generator. DatabaseManager.save_candidates implementation
            # iterates over it, which is memory safe (streaming).

            # However, DatabaseManager.save_candidates type hint asks for `list`.
            # I should update DatabaseManager to accept Iterable.
            # But based on my read of database.py, it took a list.
            # I will cast to list carefully or check database.py again.
            # database.py: def save_candidates(self, candidates: list[tuple[Atoms, dict[str, Any]]])

            # To be safe regarding OOM, I should chunk this generator.

            CHUNK_SIZE = 1000
            chunk = []
            for item in prepared_batch:
                chunk.append(item)
                if len(chunk) >= CHUNK_SIZE:
                    self.db.save_candidates(chunk)
                    chunk = []

            if chunk:
                self.db.save_candidates(chunk)

            logger.info("Batch creation of candidates completed.")

        except Exception as e:
            logger.error(f"Failed to create candidates batch: {e}")
            raise
