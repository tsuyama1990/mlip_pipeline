import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import ase.db
import numpy as np
from ase import Atoms

from mlip_autopipec.exceptions import DatabaseError

if TYPE_CHECKING:
    from mlip_autopipec.config.models import SystemConfig

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Handles the connection lifecycle to the ASE/SQLite database.
    """
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._connection: ase.db.core.Database | None = None

    def connect(self) -> ase.db.core.Database:
        """
        Establishes and returns the database connection.
        """
        if self._connection is not None:
            return self._connection

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = ase.db.connect(str(self.db_path))
            # Verify validity
            self._connection.count()
            return self._connection
        except OSError as e:
            logger.error(f"FileSystem error initializing database at {self.db_path}: {e}")
            raise DatabaseError(
                f"FileSystem error initializing database at {self.db_path}: {e}"
            ) from e
        except sqlite3.DatabaseError as e:
            logger.error(f"File at {self.db_path} is not a valid SQLite database: {e}")
            raise DatabaseError(
                f"File at {self.db_path} is not a valid SQLite database: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    def close(self) -> None:
        """Closes the active connection."""
        self._connection = None


class DatabaseManager:
    """
    Wrapper around ase.db to enforce schema and metadata requirements.

    This class manages data access and persistence operations.
    It implements the Context Manager protocol.
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the DatabaseManager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.connector = DatabaseConnector(db_path)

    def __enter__(self) -> Self:
        """Context manager entry. Ensures connection is ready."""
        self.connector.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Context manager exit. Closes connection."""
        self.connector.close()

    @property
    def _connection(self) -> ase.db.core.Database:
        """Internal helper to get connection."""
        return self.connector.connect()

    def add_structure(self, atoms: Atoms, metadata: dict[str, Any]) -> int:
        """
        Inserts an atom with metadata.

        Args:
            atoms: ASE Atoms object.
            metadata: Dictionary containing 'status', 'config_type', 'generation'.

        Returns:
            The integer ID of the inserted row.

        Raises:
            DatabaseError: If insertion fails.
        """
        try:
            id = self._connection.write(atoms, **metadata)
            return id
        except KeyError as e:
            logger.error(f"Invalid key in metadata: {e}")
            raise DatabaseError(f"Invalid key in metadata: {e}") from e
        except Exception as e:
            logger.error(f"Failed to add structure: {e}")
            raise DatabaseError(f"Failed to add structure: {e}") from e

    def count(self, selection: str | None = None, **kwargs: Any) -> int:
        """
        Counts rows matching selection.

        Args:
            selection: Raw selection string (use with caution).
            **kwargs: Parameterized selection criteria (preferred).

        Returns:
            Number of matching rows.
        """
        try:
            return self._connection.count(selection=selection, **kwargs)
        except Exception as e:
            logger.error(f"Failed to count rows: {e}")
            raise DatabaseError(f"Failed to count rows: {e}") from e

    def update_status(self, id: int, status: str) -> None:
        """
        Updates the status of a specific row.

        Args:
            id: Database ID of the row.
            status: New status string.
        """
        try:
            self._connection.update(id, status=status)
        except KeyError as e:
            logger.error(f"ID {id} not found: {e}")
            raise DatabaseError(f"ID {id} not found: {e}") from e
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            raise DatabaseError(f"Failed to update status: {e}") from e

    def update_metadata(self, id: int, data: dict[str, Any]) -> None:
        """
        Updates metadata for a specific row.

        Args:
            id: Database ID of the row.
            data: Dictionary of key-value pairs to update.
        """
        try:
            self._connection.update(id, **data)
        except KeyError as e:
            logger.error(f"ID {id} not found: {e}")
            raise DatabaseError(f"ID {id} not found: {e}") from e
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            raise DatabaseError(f"Failed to update metadata: {e}") from e

    def get_atoms(self, selection: str | None = None, **kwargs: Any) -> list[Atoms]:
        """
        Retrieve atoms matching selection.

        Args:
            selection: Selection string.
            **kwargs: Parameterized query arguments.

        Returns:
            List of ASE Atoms objects with populated info dictionary.
        """
        try:
            rows = self._connection.select(selection=selection, **kwargs)
            atoms_list = []
            for row in rows:
                at = row.toatoms()
                if hasattr(row, "key_value_pairs"):
                    at.info.update(row.key_value_pairs)
                atoms_list.append(at)
            return atoms_list
        except Exception as e:
            logger.error(f"Failed to get atoms: {e}")
            raise DatabaseError(f"Failed to get atoms: {e}") from e

    def get_entries(self, selection: str | None = None, **kwargs: Any) -> list[tuple[int, Atoms]]:
        """
        Retrieve entries as (id, Atoms) tuples.

        Args:
            selection: Selection string.
            **kwargs: Parameterized query arguments.

        Returns:
            List of (id, Atoms) tuples.
        """
        try:
            rows = self._connection.select(selection=selection, **kwargs)
            return [(row.id, row.toatoms()) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get entries: {e}")
            raise DatabaseError(f"Failed to get entries: {e}") from e

    def save_candidate(self, atoms: Atoms, metadata: dict[str, Any]) -> None:
        """
        Save a candidate structure.
        Caller is responsible for providing all necessary metadata.
        """
        self.add_structure(atoms, metadata)

    def save_dft_result(self, atoms: Atoms, result: Any, metadata: dict[str, Any]) -> None:
        """
        Save a DFT result.

        Args:
            atoms: The Atoms object (updated with results).
            result: The DFTResult object.
            metadata: Additional metadata to save.
        """
        try:
            if hasattr(result, "energy"):
                atoms.info["energy"] = result.energy
            if hasattr(result, "forces"):
                atoms.arrays["forces"] = np.array(result.forces)
            if hasattr(result, "stress"):
                atoms.info["stress"] = np.array(result.stress)

            atoms.info.update(metadata)

            self._connection.write(
                atoms, data=result.model_dump() if hasattr(result, "model_dump") else {}
            )
        except AttributeError as e:
            logger.error(f"Invalid DFTResult object: {e}")
            raise DatabaseError(f"Invalid DFTResult object: {e}") from e
        except Exception as e:
            logger.error(f"Failed to save DFT result: {e}")
            raise DatabaseError(f"Failed to save DFT result: {e}") from e

    def set_system_config(self, config: "SystemConfig") -> None:
        """Store system config in metadata."""
        try:
            self._connection.metadata = config.model_dump(mode="json")
        except Exception as e:
            logger.warning(f"Failed to store SystemConfig: {e}")

    def get_system_config(self) -> "SystemConfig":
        """Retrieve system config from metadata."""
        from pydantic import ValidationError

        from mlip_autopipec.config.models import SystemConfig

        try:
            return SystemConfig.model_validate(self._connection.metadata)
        except ValidationError as e:
            logger.error(f"Stored SystemConfig is invalid: {e}")
            raise DatabaseError(f"Stored SystemConfig is invalid: {e}") from e
        except Exception as e:
            # Catch AttributeError if metadata is missing or None
            raise DatabaseError(f"No SystemConfig found or error retrieving: {e}") from e
