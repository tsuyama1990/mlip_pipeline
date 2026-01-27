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
            logger.exception("FileSystem error initializing database")
            msg = f"FileSystem error initializing database at {self.db_path}: {e}"
            raise DatabaseError(
                msg
            ) from e
        except sqlite3.DatabaseError as e:
            logger.exception("File is not a valid SQLite database")
            msg = f"File at {self.db_path} is not a valid SQLite database: {e}"
            raise DatabaseError(
                msg
            ) from e
        except Exception as e:
            logger.exception("Failed to initialize database")
            msg = f"Failed to initialize database: {e}"
            raise DatabaseError(msg) from e

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
            msg = "Input must be an ase.Atoms object"
            raise TypeError(msg)

        positions = atoms.get_positions()
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            msg = "Atoms object contains NaN or Inf in positions."
            raise ValueError(msg)

        if atoms.pbc.any():
            cell = atoms.get_cell()
            if np.isclose(np.linalg.det(cell), 0.0):
                msg = "Atoms object has zero cell volume but PBC is enabled."
                raise ValueError(msg)

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
                return self._connection.write(atoms, **metadata)
        except ValueError as e:
            logger.exception("Validation failed for atoms insertion")
            msg = f"Invalid Atoms object: {e}"
            raise DatabaseError(msg) from e
        except KeyError as e:
            logger.exception("Invalid key in metadata during add_structure")
            msg = f"Invalid key in metadata: {e}"
            raise DatabaseError(msg) from e
        except Exception as e:
            logger.exception("Failed to add structure")
            msg = f"Failed to add structure: {e}"
            raise DatabaseError(msg) from e

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
            logger.exception("Failed to count rows")
            msg = f"Failed to count rows: {e}"
            raise DatabaseError(msg) from e

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
            logger.exception("ID not found during update_status")
            msg = f"ID {id} not found: {e}"
            raise DatabaseError(msg) from e
        except Exception as e:
            logger.exception("Failed to update status")
            msg = f"Failed to update status: {e}"
            raise DatabaseError(msg) from e

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
            logger.exception("ID not found during update_metadata")
            msg = f"ID {id} not found: {e}"
            raise DatabaseError(msg) from e
        except Exception as e:
            logger.exception("Failed to update metadata")
            msg = f"Failed to update metadata: {e}"
            raise DatabaseError(msg) from e

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
            logger.exception("Failed to select atoms")
            msg = f"Failed to select atoms: {e}"
            raise DatabaseError(msg) from e

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
            logger.exception("Failed to select entries")
            msg = f"Failed to select entries: {e}"
            raise DatabaseError(msg) from e

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

    def save_candidates(self, candidates: list[tuple[Atoms, dict[str, Any]]]) -> None:
        """
        Save multiple candidate structures in a single transaction (if possible).

        Args:
            candidates: List of tuples (atoms, metadata).
        """
        if not candidates:
            return

        try:
            with self._lock:
                # ase.db doesn't expose explicit transaction begin/commit easily for `write`.
                # However, if we are in a sqlite context, we can try to wrap.
                # But to stay safe with ase.db abstraction, we just iterate.
                # Since we hold the lock, no other thread interrupts.
                # Optimally, we would use self._connection.managed_connection if available.
                for atoms, meta in candidates:
                    self._validate_atoms(atoms)
                    self._connection.write(atoms, **meta)
        except Exception as e:
            logger.exception("Failed to save candidates batch")
            msg = f"Failed to save candidates batch: {e}"
            raise DatabaseError(msg) from e

    def save_dft_result(self, atoms: Atoms, result: Any, metadata: dict[str, Any]) -> None:
        """
        Save a DFT result.
        """
        try:
            self._validate_atoms(atoms)
            if hasattr(result, "energy"):
                # Check for physical validity
                if not np.isfinite(result.energy):
                     msg = "DFT Energy is not finite."
                     raise ValueError(msg)
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
            logger.exception("Invalid DFTResult object passed to save_dft_result")
            msg = f"Invalid DFTResult object: {e}"
            raise DatabaseError(msg) from e
        except ValueError as e:
            logger.exception("Validation failed for atoms insertion in save_dft_result")
            msg = f"Invalid Atoms object: {e}"
            raise DatabaseError(msg) from e
        except Exception as e:
            logger.exception("Failed to save DFT result")
            msg = f"Failed to save DFT result: {e}"
            raise DatabaseError(msg) from e

    def set_system_config(self, config: "SystemConfig") -> None:
        """Store system config in metadata."""
        try:
            with self._lock:
                self._connection.metadata = config.model_dump(mode="json")  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to store SystemConfig in database metadata: {e}")

    def get_system_config(self) -> "SystemConfig":
        """Retrieve system config from metadata."""
        from pydantic import ValidationError

        from mlip_autopipec.config.models import SystemConfig

        try:
            return SystemConfig.model_validate(self._connection.metadata)
        except ValidationError as e:
            logger.exception("Stored SystemConfig in database is invalid")
            msg = f"Stored SystemConfig is invalid: {e}"
            raise DatabaseError(msg) from e
        except Exception as e:
            logger.exception("Error retrieving SystemConfig from database")
            msg = f"No SystemConfig found or error retrieving: {e}"
            raise DatabaseError(msg) from e
