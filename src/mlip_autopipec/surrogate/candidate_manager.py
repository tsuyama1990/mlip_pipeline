import logging

import numpy as np
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
            msg = f"Expected ase.Atoms, got {type(atoms)}"
            raise TypeError(msg)

        # Validation logic
        pos = atoms.get_positions()
        if np.isnan(pos).any() or np.isinf(pos).any():
             msg = "Atoms object contains NaN or Inf in positions."
             raise ValueError(msg)

        self.db.save_candidates([atoms], 0, "generation")
