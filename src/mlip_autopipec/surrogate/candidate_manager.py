from typing import Any
from ase import Atoms
from mlip_autopipec.orchestration.database import DatabaseManager

class CandidateManager:
    """
    Manages candidate data persistence logic.
    Separates business logic (defaults, validation) from raw DB access.
    """
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def create_candidate(self, atoms: Atoms, metadata: dict[str, Any] | None = None) -> None:
        if metadata is None:
            metadata = {}
        # Apply defaults
        meta = metadata.copy()
        if "status" not in meta:
            meta["status"] = "pending"
        if "generation" not in meta:
            meta["generation"] = 0
        if "config_type" not in meta:
            meta["config_type"] = "candidate"

        self.db.add_structure(atoms, meta)
