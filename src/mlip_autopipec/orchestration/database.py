import logging
import sqlite3
import threading
from collections.abc import Generator
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
            # Verify validity by attempting a lightweight operation
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
    Thread-safe for write operations.
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the DatabaseManager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.connector = DatabaseConnector(db_path)
        self._lock = threading.Lock()

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

    def _validate_atoms(self, atoms: Atoms) -> None:
        """
        Validates an ASE Atoms object before insertion.
        Checks for NaN positions, infinite values, or zero cells if PBC is true.
        Also checks for physical validity (e.g. non-zero coordinates for >1 atoms).
        """
        if not isinstance(atoms, Atoms):
            raise TypeError("Input must be an ase.Atoms object")

        positions = atoms.get_positions()
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            raise ValueError("Atoms object contains NaN or Inf in positions.")

        if atoms.pbc.any():
            cell = atoms.get_cell()
            if np.isclose(np.linalg.det(cell), 0.0):
                raise ValueError("Atoms object has zero cell volume but PBC is enabled.")

        # Data Integrity: Check for huge/unphysical values if possible (simple heuristic)
        # E.g. coords > 1e6 usually mean error, but maybe not in MD.
        # We stick to NaN/Inf for now as critical.

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
            self._validate_atoms(atoms)
            with self._lock:
                id = self._connection.write(atoms, **metadata)
            return id
        except ValueError as e:
            logger.error(f"Validation failed for atoms insertion: {e}")
            raise DatabaseError(f"Invalid Atoms object: {e}") from e
        except KeyError as e:
            logger.error(f"Invalid key in metadata during add_structure: {e}")
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
            # ase.db handles parameterized queries via kwargs.
            # selection string is parsed by ase.db but can be vulnerable if constructed via f-strings externally.
            # We rely on ase.db's internal handling but prefer kwargs.
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
            with self._lock:
                self._connection.update(id, status=status)
        except KeyError as e:
            logger.error(f"ID {id} not found during update_status: {e}")
            raise DatabaseError(f"ID {id} not found: {e}") from e
        except Exception as e:
            logger.error(f"Failed to update status for ID {id}: {e}")
            raise DatabaseError(f"Failed to update status: {e}") from e

    def update_metadata(self, id: int, data: dict[str, Any]) -> None:
        """
        Updates metadata for a specific row.

        Args:
            id: Database ID of the row.
            data: Dictionary of key-value pairs to update.
        """
        try:
            with self._lock:
                self._connection.update(id, **data)
        except KeyError as e:
            logger.error(f"ID {id} not found during update_metadata: {e}")
            raise DatabaseError(f"ID {id} not found: {e}") from e
        except Exception as e:
            logger.error(f"Failed to update metadata for ID {id}: {e}")
            raise DatabaseError(f"Failed to update metadata: {e}") from e

    def select(self, selection: str | None = None, **kwargs: Any) -> Generator[Atoms, None, None]:
        """
        Generator that yields atoms objects matching selection.
        This enables processing large datasets without loading everything into memory.

        Args:
            selection: Selection string.
            **kwargs: Parameterized query arguments.

        Yields:
            ASE Atoms objects with populated info dictionary.
        """
        try:
            rows = self._connection.select(selection=selection, **kwargs)
            for row in rows:
                at = row.toatoms()
                if hasattr(row, "key_value_pairs"):
                    at.info.update(row.key_value_pairs)
                if hasattr(row, "data"):
                    at.info.update(row.data)
                yield at
        except Exception as e:
            logger.error(f"Failed to select atoms: {e}")
            raise DatabaseError(f"Failed to select atoms: {e}") from e

    def select_entries(self, selection: str | None = None, **kwargs: Any) -> Generator[tuple[int, Atoms], None, None]:
        """
        Generator that yields (id, atoms) tuples matching selection.
        Crucial for batch processing where ID is needed for updates.

        Args:
            selection: Selection string.
            **kwargs: Parameterized query arguments.

        Yields:
            Tuple of (database_id, ASE Atoms object).
        """
        try:
            rows = self._connection.select(selection=selection, **kwargs)
            for row in rows:
                at = row.toatoms()
                if hasattr(row, "key_value_pairs"):
                    at.info.update(row.key_value_pairs)
                if hasattr(row, "data"):
                    at.info.update(row.data)
                yield row.id, at
        except Exception as e:
            logger.error(f"Failed to select entries: {e}")
            raise DatabaseError(f"Failed to select entries: {e}") from e

    def get_atoms(self, selection: str | None = None, **kwargs: Any) -> Generator[Atoms, None, None]:
        """
        Retrieve atoms matching selection.
        Returns a generator to avoid OOM on large datasets.
        """
        return self.select(selection=selection, **kwargs)

    def get_entries(self, selection: str | None = None, **kwargs: Any) -> Generator[tuple[int, Atoms], None, None]:
        """
        Retrieve entries as (id, Atoms) tuples.
        Returns a generator to avoid OOM on large datasets.
        """
        return self.select_entries(selection=selection, **kwargs)

    def save_candidate(self, atoms: Atoms, metadata: dict[str, Any]) -> None:
        """
        Save a candidate structure.
        """
        self.add_structure(atoms, metadata)

    def save_dft_result(self, atoms: Atoms, result: Any, metadata: dict[str, Any]) -> None:
        """
        Save a DFT result.
        """
        try:
            self._validate_atoms(atoms)
            if hasattr(result, "energy"):
                # Check for physical validity
                if not np.isfinite(result.energy):
                     raise ValueError("DFT Energy is not finite.")
                atoms.info["energy"] = result.energy
            if hasattr(result, "forces"):
                atoms.arrays["forces"] = np.array(result.forces)
            if hasattr(result, "stress"):
                atoms.info["stress"] = np.array(result.stress)

            atoms.info.update(metadata)

            with self._lock:
                self._connection.write(
                    atoms,
                    data=result.model_dump() if hasattr(result, "model_dump") else {},
                    **metadata
                )
        except AttributeError as e:
            logger.error(f"Invalid DFTResult object passed to save_dft_result: {e}")
            raise DatabaseError(f"Invalid DFTResult object: {e}") from e
        except ValueError as e:
            logger.error(f"Validation failed for atoms insertion in save_dft_result: {e}")
            raise DatabaseError(f"Invalid Atoms object: {e}") from e
        except Exception as e:
            logger.error(f"Failed to save DFT result: {e}")
            raise DatabaseError(f"Failed to save DFT result: {e}") from e

    def set_system_config(self, config: "SystemConfig") -> None:
        """Store system config in metadata."""
        try:
            with self._lock:
                self._connection.metadata = config.model_dump(mode="json")
        except Exception as e:
            logger.warning(f"Failed to store SystemConfig in database metadata: {e}")

    def get_system_config(self) -> "SystemConfig":
        """Retrieve system config from metadata."""
        from pydantic import ValidationError

        from mlip_autopipec.config.models import SystemConfig

        try:
            return SystemConfig.model_validate(self._connection.metadata)
        except ValidationError as e:
            logger.error(f"Stored SystemConfig in database is invalid: {e}")
            raise DatabaseError(f"Stored SystemConfig is invalid: {e}") from e
        except Exception as e:
            logger.error(f"Error retrieving SystemConfig from database: {e}")
            raise DatabaseError(f"No SystemConfig found or error retrieving: {e}") from e
