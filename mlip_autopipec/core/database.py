from pathlib import Path
from typing import Dict, List, Any
import contextlib
import ase.db
from ase import Atoms
import numpy as np

class DatabaseManager:
    """
    Wrapper around ase.db for managing atomic structures and calculations.
    """
    def __init__(self, db_path: Path) -> None:
        """
        Initialize the database manager.
        Establishes connection to the SQLite database. Creates it if it doesn't exist.
        """
        self.db_path = db_path
        self._connection = ase.db.connect(str(db_path))
        # Ensure database file is created
        with contextlib.suppress(Exception):
            self._connection.count()

    def count(self, **kwargs: Any) -> int:
        """Returns the number of rows matching the query."""
        # ase.db.count returns int
        return self._connection.count(**kwargs)

    def add_calculation(self, atoms: Atoms, metadata: Dict[str, Any]) -> int:
        """
        Adds a completed calculation to the database.

        Args:
            atoms: Atoms object with calculated energy and forces.
            metadata: Dictionary of metadata to store.

        Returns:
            The ID of the inserted row.

        Raises:
            ValueError: If energy or forces are missing.
        """
        # Validate existence of energy and forces
        try:
            _ = atoms.get_potential_energy()
            forces = atoms.get_forces()
        except Exception as e:
            # Spec requirement: atoms object must have results
            msg = "Atoms object must have energy and forces available."
            raise ValueError(msg) from e

        if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
            msg = "Forces contain NaN or Inf values."
            raise ValueError(msg)

        # Flatten metadata for key-value pairs (columns)
        # We assume metadata is relatively flat or we store complex objects in 'data'
        # Treat all metadata as key-value pairs for `write` (kwargs).
        return self._connection.write(atoms, **metadata)

    def save_candidate(self, atoms: Atoms, metadata: Dict[str, Any]) -> int:
        """
        Saves a candidate structure (without calculation results) with 'pending' status.
        """
        data = metadata.copy()
        data["status"] = "pending"
        return self._connection.write(atoms, **data)

    def get_pending_calculations(self) -> List[Atoms]:
        """Retrieves entries flagged for computation (status='pending')."""
        # select returns an iterator of Rows. Row.toatoms() converts to Atoms.
        rows = self._connection.select(status="pending")
        return [row.toatoms() for row in rows]
