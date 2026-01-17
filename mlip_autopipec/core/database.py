import ase.db
from pathlib import Path
from typing import Any, Dict
from ase import Atoms
from mlip_autopipec.config.schemas.system import SystemConfig

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self, system_config: SystemConfig) -> None:
        """
        Initializes the database and stores the system configuration.
        """
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to DB
        with ase.db.connect(self.db_path) as db:
            # Force initialization of the database file structure
            # This is a quirk of ase.db with SQLite
            db.count()

            # Prepare metadata
            # Convert SystemConfig to a JSON-compatible dictionary
            config_dict = system_config.model_dump(mode='json')

            # Store configuration in metadata
            # We overwrite existing metadata to ensure it matches current config
            db.metadata = config_dict

    def get_metadata(self) -> Dict[str, Any]:
        """Retrieves the database metadata."""
        # Check if file exists to avoid creating an empty one just for reading?
        # ase.db.connect will create it. But get_metadata on empty non-existent DB might differ.
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file {self.db_path} does not exist.")

        with ase.db.connect(self.db_path) as db:
            return db.metadata

    def add_structure(self, atoms: Atoms, **kwargs: Any) -> int:
        """
        Adds an atomic structure to the database.
        Returns the ID of the inserted row.
        """
        with ase.db.connect(self.db_path) as db:
            return db.write(atoms, **kwargs)
