from pathlib import Path
from typing import Any

import ase.db

from mlip_autopipec.config.models import SystemConfig


class DatabaseManager:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self, config: SystemConfig) -> None:
        """
        Initializes the database with metadata from the SystemConfig.
        Handles the case where the file does not exist.
        """
        # Ensure parent directory exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dict for metadata
        # Pydantic models need to be dumped to standard python types for ase.db (json serialization)
        # using mode='json' to ensure paths are converted to strings, etc.
        metadata = config.model_dump(mode="json")

        with ase.db.connect(self.db_path) as conn:
            # Check if already initialized to avoid overwriting if we want that behavior
            # But spec says: "initialize(): Checks if file exists. If not, calls ase.db.connect, and writes a metadata key"
            # It also implies we should probably merge or update if it exists, or just set it.
            # UAT says "Verify that the initialized database contains the configuration settings in its metadata."

            # Write metadata
            conn.metadata = metadata
            # Force initialization/write by counting (or just writing metadata might be enough, but memory says otherwise)
            # Accessing conn.metadata ... immediately after creation can raise ... using conn.count() ... forces initialization
            # However, we are WRITING metadata here. But to be safe and ensure the file is created:
            conn.count()

    def get_metadata(self) -> dict[str, Any]:
        with ase.db.connect(self.db_path) as conn:
            # Ensure initialization before reading metadata if strictly necessary, but usually connect reads it.
            # Memory says accessing metadata *immediately after creation* fails. Here we assume it exists.
            if not self.db_path.exists():
                return {}
            return dict(conn.metadata)
