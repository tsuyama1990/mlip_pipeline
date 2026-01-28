import logging
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect

from mlip_autopipec.config.schemas.core import SystemConfig

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations."""


class DatabaseManager:
    """
    Manages interaction with the ASE SQLite database.
    Implements streaming for scalable data retrieval.
    Thread-safe for basic operations.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._connection = None
        self._initialize()

    def _initialize(self) -> None:
        """Initializes the database connection."""
        try:
            self._connection = connect(str(self.db_path))
            self._connection.count()
        except Exception as e:
             logger.exception(f"Failed to initialize database at {self.db_path}")
             raise DatabaseError(f"Failed to initialize database: {e}") from e

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _validate_atoms(self, atoms: Atoms) -> None:
        """Validates Atoms object integrity before insertion."""
        if not isinstance(atoms, Atoms):
            raise TypeError("Input must be an ase.Atoms object")

        positions = atoms.get_positions()
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            raise ValueError("Atoms object contains NaN or Inf in positions.")

        if atoms.pbc.any():
            cell = atoms.get_cell()
            if np.isclose(np.linalg.det(cell), 0.0):
                raise ValueError("Atoms object has zero cell volume but PBC is enabled.")

    def add_structure(self, atoms: Atoms, metadata: dict[str, Any] | None = None) -> int:
        metadata = metadata or {}
        try:
            self._validate_atoms(atoms)
            with self._lock:
                row_id = self._connection.write(atoms, **metadata)
            return row_id
        except Exception as e:
            logger.exception(f"Failed to add structure: {e}")
            raise DatabaseError(f"Failed to add structure: {e}") from e

    def count(self, selection: str | None = None, **kwargs: Any) -> int:
        try:
            return self._connection.count(selection=selection, **kwargs)
        except Exception as e:
            logger.exception("Failed to count rows")
            raise DatabaseError(f"Failed to count rows: {e}") from e

    def update_status(self, row_id: int, status: str) -> None:
        try:
            with self._lock:
                self._connection.update(row_id, status=status)
        except Exception as e:
            logger.exception(f"Failed to update status for ID {row_id}")
            raise DatabaseError(f"Failed to update status: {e}") from e

    def update_metadata(self, row_id: int, data: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._connection.update(row_id, **data)
        except Exception as e:
            logger.exception(f"Failed to update metadata for ID {row_id}")
            raise DatabaseError(f"Failed to update metadata: {e}") from e

    def select(self, selection: str | None = None, **kwargs: Any) -> Generator[Atoms, None, None]:
        """
        Streams atoms objects matching the selection.
        """
        try:
            for row in self._connection.select(selection=selection, **kwargs):
                at = row.toatoms()
                at.info.update(row.key_value_pairs)
                at.info["_db_id"] = row.id
                yield at
        except Exception as e:
            logger.exception("Failed to select atoms")
            raise DatabaseError(f"Failed to select atoms: {e}") from e

    def select_entries(
        self, selection: str | None = None, **kwargs: Any
    ) -> Generator[tuple[int, Atoms], None, None]:
        """
        Streams (id, Atoms) tuples.
        """
        try:
            for row in self._connection.select(selection=selection, **kwargs):
                at = row.toatoms()
                at.info.update(row.key_value_pairs)
                yield row.id, at
        except Exception as e:
            logger.exception("Failed to select entries")
            raise DatabaseError(f"Failed to select entries: {e}") from e

    def get_atoms(
        self, selection: str | None = None, limit: int | None = None
    ) -> Generator[Atoms, None, None]:
        yield from self.select(selection, limit=limit)

    def save_candidates(self, candidates: list[Atoms], cycle_index: int, method: str) -> None:
        meta = {"cycle": cycle_index, "origin": method, "status": "candidate", "converged": False}
        try:
            with self._lock:
                for atoms in candidates:
                    self._validate_atoms(atoms)
                    self._connection.write(atoms, **meta)
        except Exception as e:
            logger.exception("Failed to save candidates batch")
            raise DatabaseError(f"Failed to save candidates batch: {e}") from e

    def save_dft_result(self, atoms: Atoms, result: Any, metadata: dict[str, Any]) -> None:
        try:
            if not np.isfinite(result.energy):
                raise ValueError("DFT Energy is not finite.")

            atoms.info["energy"] = result.energy

            if hasattr(result, "forces") and result.forces is not None:
                f = np.array(result.forces)
                if f.shape != (len(atoms), 3):
                    raise ValueError(f"Forces shape mismatch: {f.shape} vs ({len(atoms)}, 3)")
                atoms.new_array("forces", f)

            if hasattr(result, "stress") and result.stress is not None:
                s = np.array(result.stress)
                atoms.info["stress"] = s

            meta = metadata.copy()
            meta["converged"] = result.converged
            if not result.converged:
                meta["error"] = result.error_message

            self.add_structure(atoms, meta)

        except Exception as e:
            logger.exception("Failed to save DFT result")
            raise DatabaseError(f"Failed to save DFT result: {e}") from e

    def set_system_config(self, config: "SystemConfig") -> None:
        try:
            with self._lock:
                self._connection.metadata = config.model_dump()
        except Exception as e:
            logger.exception("Failed to set system config")
            raise DatabaseError(f"Failed to set system config: {e}") from e

    def get_system_config(self) -> "SystemConfig":
        try:
            return SystemConfig.model_validate(self._connection.metadata)
        except Exception as e:
            logger.exception("Error retrieving SystemConfig")
            raise DatabaseError(f"No SystemConfig found or error retrieving: {e}") from e
