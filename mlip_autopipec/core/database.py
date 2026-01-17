from pathlib import Path
from typing import Any

import ase.db
from ase.db.core import Database

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.exceptions import DatabaseError


class DatabaseManager:
    """
    Wrapper around ase.db to enforce schema and metadata requirements.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._connection: Database | None = None

    def initialize(self) -> None:
        """
        Initializes the database connection.
        If the database does not exist, it is created.
        """
        try:
            # Ensure parent directory exists (though WorkspaceManager handles this, redundancy is safe)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect (or create)
            self._connection = ase.db.connect(str(self.db_path))

            # Force initialization
            self._connection.count()
        except Exception as e:
            msg = f"Failed to initialize database at {self.db_path}: {e}"
            raise DatabaseError(msg) from e

    def set_system_config(self, config: SystemConfig) -> None:
        """
        Stores the system configuration in the database metadata.
        This must be called during initialization or setup.
        """
        if self._connection is None:
            self.initialize()

        try:
            config_dict = config.model_dump(mode='json')

            # Update metadata
            if self._connection is None: # explicit check for mypy though initialize guarantees it
                msg = "Database connection is None after initialization."
                raise DatabaseError(msg)

            current_metadata = self._connection.metadata.copy()
            current_metadata.update(config_dict)
            self._connection.metadata = current_metadata
        except Exception as e:
            msg = "Failed to write system configuration to database metadata."
            raise DatabaseError(msg) from e

    def get_metadata(self) -> dict[str, Any]:
        """
        Retrieves the metadata from the database.
        """
        if self._connection is None:
            self.initialize()

        if self._connection is None:
             msg = "Database connection is None after initialization."
             raise DatabaseError(msg)

        return self._connection.metadata # type: ignore
