"""
Core database functionality for the MLIP-AutoPipe project.
Wraps ase.db to ensure schema consistency and provenance tracking.
"""
import ase.db
from pathlib import Path
from typing import Any, Dict
from ase import Atoms
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.exceptions import DatabaseError

class DatabaseManager:
    """
    Manages the connection to the ASE database (SQLite).
    Enforces metadata schemas and handles initialization.
    """
    def __init__(self, db_path: Path):
        """
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path

    def initialize(self, system_config: SystemConfig) -> None:
        """
        Initializes the database and stores the system configuration.

        Args:
            system_config: The full system configuration to store as metadata.

        Raises:
            DatabaseError: If database initialization fails.
        """
        # Ensure directory exists
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise DatabaseError(f"Failed to create database directory {self.db_path.parent}: {e}") from e

        try:
            # Connect to DB
            with ase.db.connect(self.db_path) as db:
                # Force initialization of the database file structure
                db.count()

                # Prepare metadata
                config_dict = system_config.model_dump(mode='json')

                # Store configuration in metadata
                db.metadata = config_dict
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database at {self.db_path}: {e}") from e

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves the database metadata.

        Returns:
            Dictionary containing the database metadata.

        Raises:
            FileNotFoundError: If the database file does not exist.
            DatabaseError: If reading metadata fails.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file {self.db_path} does not exist.")

        try:
            with ase.db.connect(self.db_path) as db:
                return db.metadata
        except Exception as e:
            raise DatabaseError(f"Failed to read metadata from {self.db_path}: {e}") from e

    def add_structure(self, atoms: Atoms, **kwargs: Any) -> int:
        """
        Adds an atomic structure to the database.

        Args:
            atoms: The ase.Atoms object to add.
            **kwargs: Additional key-value pairs to store as data.

        Returns:
            The ID of the inserted row.

        Raises:
            DatabaseError: If the write operation fails.
        """
        try:
            with ase.db.connect(self.db_path) as db:
                return db.write(atoms, **kwargs)
        except Exception as e:
            raise DatabaseError(f"Failed to add structure to database: {e}") from e
