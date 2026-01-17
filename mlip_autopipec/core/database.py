from pathlib import Path
from typing import Any

from ase import Atoms
from ase.db import connect

from mlip_autopipec.config.models import SystemConfig


class DatabaseManager:
    """Wrapper around ase.db to enforce schema and metadata."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self, config: SystemConfig) -> None:
        """Initializes the database with system configuration metadata."""
        # Ensure the directory exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        meta = config.model_dump(mode='json')

        with connect(self.db_path) as conn:
            conn.metadata = meta

    def get_metadata(self) -> dict[str, Any]:
        """Retrieves the system configuration metadata."""
        with connect(self.db_path) as conn:
            return conn.metadata

    def add_structure(self, atoms: Atoms, **kwargs: Any) -> int:
        """
        Adds an atomic structure to the database with enforced metadata.
        Returns the ID of the inserted row.
        """
        with connect(self.db_path) as conn:
            return conn.write(atoms, **kwargs)
