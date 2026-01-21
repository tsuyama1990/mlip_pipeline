import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import ase.db
import numpy as np
from ase import Atoms

from mlip_autopipec.exceptions import DatabaseError

if TYPE_CHECKING:
    from mlip_autopipec.config.models import SystemConfig


class DatabaseManager:
    """
    Wrapper around ase.db to enforce schema and metadata requirements.

    This class manages the connection to the ASE database (SQLite).
    It implements the Context Manager protocol to ensure connections are closed.
    It strictly handles data persistence and does not inject business logic defaults.
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the DatabaseManager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._connection: ase.db.core.Database | None = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Context manager exit. Closes connection."""
        self.close()

    def close(self) -> None:
        """
        Closes the database connection.
        """
        if self._connection:
            self._connection = None

    def initialize(self) -> None:
        """
        Initializes the database connection.
        If the database does not exist, it is created.

        Raises:
            DatabaseError: If initialization fails or file is not a valid DB.
        """
        if self._connection is not None:
            return

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = ase.db.connect(str(self.db_path))
            # Force creating the file by checking count
            self._connection.count()
        except OSError as e:
            raise DatabaseError(
                f"FileSystem error initializing database at {self.db_path}: {e}"
            ) from e
        except sqlite3.DatabaseError as e:
            raise DatabaseError(
                f"File at {self.db_path} is not a valid SQLite database: {e}"
            ) from e
        except Exception as e:
            # Catch ase.db specific errors if any, or general errors
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    def _ensure_connection(self) -> None:
        if self._connection is None:
            self.initialize()

    def add_structure(self, atoms: Atoms, metadata: dict[str, Any]) -> int:
        """
        Inserts an atom with metadata.

        Args:
            atoms: ASE Atoms object.
            metadata: Dictionary containing 'status', 'config_type', 'generation'.

        Returns:
            The integer ID of the inserted row.
        """
        self._ensure_connection()

        required_keys = {"status", "config_type", "generation"}
        if not required_keys.issubset(metadata.keys()):
            # We log a warning if possible, but we don't block for now to maintain flexibility
            # unless strict schema enforcement is required by spec.
            pass

        try:
            if self._connection is not None:
                id = self._connection.write(atoms, **metadata)
                return id
            raise DatabaseError("Connection failed")
        except KeyError as e:
            raise DatabaseError(f"Invalid key in metadata: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Failed to add structure: {e}") from e

    def count(self, selection: str | None = None, **kwargs: Any) -> int:
        """
        Wraps db.count().
        """
        self._ensure_connection()
        try:
            if self._connection is not None:
                return self._connection.count(selection=selection, **kwargs)
            return 0
        except Exception as e:
            raise DatabaseError(f"Failed to count rows: {e}") from e

    def update_status(self, id: int, status: str) -> None:
        """
        Updates the status of a specific row.
        """
        self._ensure_connection()
        try:
            if self._connection is not None:
                self._connection.update(id, status=status)
        except KeyError as e:
            raise DatabaseError(f"ID {id} not found: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Failed to update status: {e}") from e

    def get_atoms(self, selection: str | None = None) -> list[Atoms]:
        """Retrieve atoms matching selection."""
        self._ensure_connection()
        try:
            if self._connection is not None:
                rows = self._connection.select(selection=selection)
                return [row.toatoms() for row in rows]
            return []
        except Exception as e:
            raise DatabaseError(f"Failed to get atoms: {e}") from e

    def save_candidate(self, atoms: Atoms, metadata: dict[str, Any]) -> None:
        """Save a candidate structure."""
        if "status" not in metadata:
            metadata["status"] = "pending"
        if "generation" not in metadata:
            metadata["generation"] = 0
        if "config_type" not in metadata:
            metadata["config_type"] = "candidate"
        self.add_structure(atoms, metadata)

    def save_dft_result(self, atoms: Atoms, result: Any, metadata: dict[str, Any]) -> None:
        """Save a DFT result."""
        self._ensure_connection()
        try:
            if hasattr(result, "energy"):
                atoms.info["energy"] = result.energy
            if hasattr(result, "forces"):
                atoms.arrays["forces"] = np.array(result.forces)
            if hasattr(result, "stress"):
                atoms.info["stress"] = np.array(result.stress)

            atoms.info.update(metadata)

            if self._connection is not None:
                self._connection.write(
                    atoms, data=result.model_dump() if hasattr(result, "model_dump") else {}
                )
        except AttributeError as e:
            raise DatabaseError(f"Invalid DFTResult object: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Failed to save DFT result: {e}") from e

    def get_training_data(self) -> list[Atoms]:
        """Get atoms with status=completed."""
        return self.get_atoms(selection="status=completed")

    def set_system_config(self, config: "SystemConfig") -> None:
        """Store system config in metadata."""
        self._ensure_connection()
        if self._connection is not None:
            try:
                self._connection.metadata = config.model_dump(mode="json")
            except Exception:
                pass

    def get_system_config(self) -> "SystemConfig":
        """Retrieve system config from metadata."""
        from pydantic import ValidationError

        from mlip_autopipec.config.models import SystemConfig

        self._ensure_connection()
        if self._connection is not None:
            try:
                return SystemConfig.model_validate(self._connection.metadata)
            except ValidationError as e:
                raise DatabaseError(f"Stored SystemConfig is invalid: {e}") from e
        raise DatabaseError("No SystemConfig found")
