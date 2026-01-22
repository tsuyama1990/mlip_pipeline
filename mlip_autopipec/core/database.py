import sqlite3
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import ase.db
import numpy as np
from ase import Atoms

from mlip_autopipec.exceptions import DatabaseError

if TYPE_CHECKING:
    from mlip_autopipec.config.models import SystemConfig

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Wrapper around ase.db to enforce schema and metadata requirements.

    This class manages the connection to the ASE database (SQLite).
    It implements the Context Manager protocol to ensure connections are closed.
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

        try:
            if self._connection is not None:
                id = self._connection.write(atoms, **metadata)
                return id
            raise DatabaseError("Connection failed")
        except KeyError as e:
            logger.error(f"Invalid key in metadata: {e}")
            raise DatabaseError(f"Invalid key in metadata: {e}") from e
        except Exception as e:
            logger.error(f"Failed to add structure: {e}")
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
            logger.error(f"Failed to count rows: {e}")
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
            logger.error(f"ID {id} not found: {e}")
            raise DatabaseError(f"ID {id} not found: {e}") from e
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            raise DatabaseError(f"Failed to update status: {e}") from e

    def update_metadata(self, id: int, data: dict[str, Any]) -> None:
        """
        Updates metadata for a specific row.
        """
        self._ensure_connection()
        try:
            if self._connection is not None:
                self._connection.update(id, **data)
        except KeyError as e:
            logger.error(f"ID {id} not found: {e}")
            raise DatabaseError(f"ID {id} not found: {e}") from e
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            raise DatabaseError(f"Failed to update metadata: {e}") from e

    def get_atoms(self, selection: str | None = None) -> list[Atoms]:
        """Retrieve atoms matching selection."""
        self._ensure_connection()
        try:
            if self._connection is not None:
                rows = self._connection.select(selection=selection)
                atoms_list = []
                for row in rows:
                    at = row.toatoms()
                    if hasattr(row, "key_value_pairs"):
                        at.info.update(row.key_value_pairs)
                    atoms_list.append(at)
                return atoms_list
            return []
        except Exception as e:
            logger.error(f"Failed to get atoms: {e}")
            raise DatabaseError(f"Failed to get atoms: {e}") from e

    def get_entries(self, selection: str | None = None) -> list[tuple[int, Atoms]]:
        """
        Retrieve entries as (id, Atoms) tuples.
        """
        self._ensure_connection()
        try:
            if self._connection is not None:
                rows = self._connection.select(selection=selection)
                return [(row.id, row.toatoms()) for row in rows]
            return []
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
            logger.error(f"Invalid DFTResult object: {e}")
            raise DatabaseError(f"Invalid DFTResult object: {e}") from e
        except Exception as e:
            logger.error(f"Failed to save DFT result: {e}")
            raise DatabaseError(f"Failed to save DFT result: {e}") from e

    def set_system_config(self, config: "SystemConfig") -> None:
        """Store system config in metadata."""
        self._ensure_connection()
        if self._connection is not None:
            try:
                self._connection.metadata = config.model_dump(mode="json")
            except Exception as e:
                logger.warning(f"Failed to store SystemConfig: {e}")

    def get_system_config(self) -> "SystemConfig":
        """Retrieve system config from metadata."""
        from pydantic import ValidationError

        from mlip_autopipec.config.models import SystemConfig

        self._ensure_connection()
        if self._connection is not None:
            try:
                return SystemConfig.model_validate(self._connection.metadata)
            except ValidationError as e:
                logger.error(f"Stored SystemConfig is invalid: {e}")
                raise DatabaseError(f"Stored SystemConfig is invalid: {e}") from e
        raise DatabaseError("No SystemConfig found")
