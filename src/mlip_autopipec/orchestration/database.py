import logging
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.exceptions import DatabaseError

logger = logging.getLogger(__name__)


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
            # Test connection
            if self._connection:
                self._connection.count()
        except Exception as e:
             logger.exception(f"Failed to initialize database at {self.db_path}")
             msg = f"Failed to initialize database: {e}"
             raise DatabaseError(msg) from e

    def close(self) -> None:
        pass

    def __enter__(self):
        if self._connection is None:
            self._initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _validate_atoms(self, atoms: Atoms) -> None:
        """Validates Atoms object integrity before insertion."""
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

    def add_structure(self, atoms: Atoms, metadata: dict[str, Any] | None = None) -> int:
        metadata = metadata or {}
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            self._validate_atoms(atoms)
            with self._lock:
                row_id = self._connection.write(atoms, **metadata)
            return row_id
        except Exception as e:
            logger.exception(f"Failed to add structure: {e}")
            msg = f"Failed to add structure: {e}"
            raise DatabaseError(msg) from e

    def count(self, selection: str | None = None, **kwargs: Any) -> int:
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            return self._connection.count(selection=selection, **kwargs)
        except Exception as e:
            logger.exception("Failed to count rows")
            msg = f"Failed to count rows: {e}"
            raise DatabaseError(msg) from e

    def update_status(self, row_id: int, status: str) -> None:
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            with self._lock:
                self._connection.update(row_id, status=status)
        except Exception as e:
            logger.exception(f"Failed to update status for ID {row_id}")
            msg = f"Failed to update status: {e}"
            raise DatabaseError(msg) from e

    def update_metadata(self, row_id: int, data: dict[str, Any]) -> None:
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            with self._lock:
                self._connection.update(row_id, **data)
        except Exception as e:
            logger.exception(f"Failed to update metadata for ID {row_id}")
            msg = f"Failed to update metadata: {e}"
            raise DatabaseError(msg) from e

    def select(self, selection: str | None = None, **kwargs: Any) -> Generator[Atoms, None, None]:
        """
        Streams atoms objects matching the selection.
        """
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            # ase.db.select returns a generator
            for row in self._connection.select(selection=selection, **kwargs):
                at = row.toatoms()
                at.info.update(row.key_value_pairs)
                at.info["_db_id"] = row.id
                yield at
        except Exception as e:
            logger.exception("Failed to select atoms")
            msg = f"Failed to select atoms: {e}"
            raise DatabaseError(msg) from e

    def select_entries(
        self, selection: str | None = None, **kwargs: Any
    ) -> Generator[tuple[int, Atoms], None, None]:
        """
        Streams (id, Atoms) tuples.
        """
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            for row in self._connection.select(selection=selection, **kwargs):
                at = row.toatoms()
                at.info.update(row.key_value_pairs)
                yield row.id, at
        except Exception as e:
            logger.exception("Failed to select entries")
            msg = f"Failed to select entries: {e}"
            raise DatabaseError(msg) from e

    def get_atoms(
        self, selection: str | None = None, limit: int | None = None
    ) -> Generator[Atoms, None, None]:
        # This implementation ensures streaming (yield from generator)
        yield from self.select(selection, limit=limit)

    def save_candidates(self, candidates: list[Atoms], cycle_index: int, method: str) -> None:
        meta = {"cycle": cycle_index, "origin": method, "status": "candidate", "converged": False}
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            with self._lock:
                for atoms in candidates:
                    self._validate_atoms(atoms)
                    self._connection.write(atoms, **meta)
        except Exception as e:
            logger.exception("Failed to save candidates batch")
            msg = f"Failed to save candidates batch: {e}"
            raise DatabaseError(msg) from e

    def save_dft_result(self, atoms: Atoms, result: Any, metadata: dict[str, Any]) -> None:
        try:
            if not np.isfinite(result.energy):
                msg = "DFT Energy is not finite."
                raise ValueError(msg)

            # Store energy in info fallback
            atoms.info["energy"] = result.energy

            forces = None
            if hasattr(result, "forces") and result.forces is not None:
                forces = np.array(result.forces)
                if forces.shape != (len(atoms), 3):
                    msg = f"Forces shape mismatch: {forces.shape} vs ({len(atoms)}, 3)"
                    raise ValueError(msg)

            stress = None
            if hasattr(result, "stress") and result.stress is not None:
                stress = np.array(result.stress)
                atoms.info["stress"] = stress

            # Attach SinglePointCalculator to properly save results
            calc = SinglePointCalculator(
                atoms,
                energy=result.energy,
                forces=forces,
                stress=stress
            )
            atoms.calc = calc

            meta = metadata.copy()
            meta["converged"] = result.converged

            if not result.converged:
                meta["error"] = result.error_message

            self.add_structure(atoms, meta)

        except Exception as e:
            logger.exception("Failed to save DFT result")
            msg = f"Failed to save DFT result: {e}"
            raise DatabaseError(msg) from e

    def set_system_config(self, config: "SystemConfig") -> None:
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            with self._lock:
                # Use mode='json' to serialize Paths to strings
                self._connection.metadata = config.model_dump(mode='json')
        except Exception as e:
            logger.exception("Failed to set system config")
            msg = f"Failed to set system config: {e}"
            raise DatabaseError(msg) from e

    def get_system_config(self) -> "SystemConfig":
        if self._connection is None:
             msg = "Database not connected"
             raise DatabaseError(msg)
        try:
            return SystemConfig.model_validate(self._connection.metadata)
        except Exception as e:
            logger.exception("Error retrieving SystemConfig")
            msg = f"No SystemConfig found or error retrieving: {e}"
            raise DatabaseError(msg) from e
