from pathlib import Path
from typing import Any

import ase.db
from ase.db.core import Database

from mlip_autopipec.config.models import SystemConfig


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
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect (or create)
        self._connection = ase.db.connect(str(self.db_path))

        # We need to perform an operation to actually create the file if it's SQLite and new
        # But ase.db might not create it until a write happens or we access metadata?
        # Accessing count() is usually safe.
        self._connection.count()

    def set_system_config(self, config: SystemConfig) -> None:
        """
        Stores the system configuration in the database metadata.
        This must be called during initialization or setup.
        """
        if self._connection is None:
            self.initialize()

        # ase.db metadata is a dictionary.
        # We store the config under specific keys.
        # SystemConfig is complex, so we dump it to dict.

        config_dict = config.model_dump(mode='json')

        # Update metadata
        # ase.db.core.Database.metadata is a property that reads/writes
        # But to update, we usually have to do:
        current_metadata = self._connection.metadata.copy()
        current_metadata.update(config_dict)
        self._connection.metadata = current_metadata

    def get_metadata(self) -> dict[str, Any]:
        """
        Retrieves the metadata from the database.
        """
        if self._connection is None:
            self.initialize()
        return self._connection.metadata # type: ignore
