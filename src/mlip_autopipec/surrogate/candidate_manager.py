import logging

from ase import Atoms

from mlip_autopipec.orchestration.database import DatabaseManager

logger = logging.getLogger(__name__)


class CandidateManager:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def create_candidate(self, atoms: Atoms, metadata: dict | None = None):
        if metadata is None:
            metadata = {}

        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected ase.Atoms, got {type(atoms)}")

        # Validation logic...

        self.db.save_candidates([atoms], 0, "generation")
