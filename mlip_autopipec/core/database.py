from pathlib import Path
from typing import Any
import ase.db
from mlip_autopipec.config.models import SystemConfig

class DatabaseManager:
    """
    Manages interactions with the ASE database (SQLite/PostgreSQL).
    Enforces schema compliance and provenance tracking.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self, config: SystemConfig) -> None:
        """
        Initializes the database.
        If the file does not exist, it is created.
        The SystemConfig is written to the database metadata for provenance.
        """
        # Ensure directory exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to DB (creates file if not exists)
        # We use a context manager to ensure connection is closed,
        # although ase.db.connect often returns a connection that needs strict management.
        # For sqlite, connect() returns a connection object.
        with ase.db.connect(self.db_path) as db:
            # Serialize the configuration
            config_dict = config.model_dump(mode='json')

            # Update metadata
            # Note: ase.db metadata handling can be tricky.
            # Usually one writes metadata during write() or sets it on the db object.
            # For a new DB, setting db.metadata works.
            # For existing, we should check/merge.

            current_metadata = db.metadata
            current_metadata.update({"system_config": config_dict})
            db.metadata = current_metadata

    def get_metadata(self) -> dict[str, Any]:
        """
        Retrieves the metadata dictionary from the database.
        """
        if not self.db_path.exists():
             raise FileNotFoundError(f"Database file not found: {self.db_path}")

        with ase.db.connect(self.db_path) as db:
            return db.metadata
