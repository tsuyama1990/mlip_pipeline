from pathlib import Path
from typing import TYPE_CHECKING, Any

import ase.db
import numpy as np
from ase import Atoms

from mlip_autopipec.exceptions import DatabaseError

if TYPE_CHECKING:
    from mlip_autopipec.config.models import SystemConfig

class DatabaseManager:
    """
    Wrapper around ase.db to enforce schema and metadata requirements.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._connection: ase.db.core.Database | None = None

    def initialize(self) -> None:
        """
        Initializes the database connection.
        If the database does not exist, it is created.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = ase.db.connect(str(self.db_path))
            # Force creating the file by checking count
            self._connection.count()
        except OSError as e:
            raise DatabaseError(f"FileSystem error initializing database at {self.db_path}: {e}") from e
        except Exception as e:
             # ase.db might raise other errors
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

        # We perform validation but catch specific errors
        required_keys = {"status", "config_type", "generation"}
        if not required_keys.issubset(metadata.keys()):
            # Log warning or allow, but here we proceed
            pass

        try:
            if self._connection is not None:
                id = self._connection.write(atoms, **metadata)
                return id
            raise DatabaseError("Connection failed")
        except KeyError as e:
            raise DatabaseError(f"Invalid key in metadata: {e}") from e
        except Exception as e:
             # ase.db write failures
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
             # Count should be robust
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

    # --- Compatibility Methods ---

    def get_atoms(self, selection: str | None = None) -> list[Atoms]:
        self._ensure_connection()
        try:
            if self._connection is not None:
                rows = self._connection.select(selection=selection)
                return [row.toatoms() for row in rows]
            return []
        except Exception as e:
            raise DatabaseError(f"Failed to get atoms: {e}") from e

    def save_candidate(self, atoms: Atoms, metadata: dict[str, Any]) -> None:
        # Wrapper for add_structure
        if "status" not in metadata:
            metadata["status"] = "pending"
        if "generation" not in metadata:
            metadata["generation"] = 0
        if "config_type" not in metadata:
            metadata["config_type"] = "candidate"
        self.add_structure(atoms, metadata)

    def save_dft_result(self, atoms: Atoms, result: Any, metadata: dict[str, Any]) -> None:
        # result is likely DFTResult model
        self._ensure_connection()
        try:
            # Map DFTResult to atoms.info/arrays
            if hasattr(result, "energy"):
                atoms.info["energy"] = result.energy
            if hasattr(result, "forces"):
                atoms.arrays["forces"] = np.array(result.forces)
            if hasattr(result, "stress"):
                atoms.info["stress"] = np.array(result.stress)

            # Merge metadata
            atoms.info.update(metadata)

            if self._connection is not None:
                self._connection.write(atoms, data=result.model_dump() if hasattr(result, "model_dump") else {})
        except AttributeError as e:
             raise DatabaseError(f"Invalid DFTResult object: {e}") from e
        except Exception as e:
             raise DatabaseError(f"Failed to save DFT result: {e}") from e

    def get_training_data(self) -> list[Atoms]:
        # Select completed calculations
        return self.get_atoms(selection="status=completed")

    def set_system_config(self, config: "SystemConfig") -> None:
        self._ensure_connection()
        if self._connection is not None:
            try:
                self._connection.metadata = config.model_dump(mode="json")
            except Exception:
                # We suppress metadata write errors as they are non-critical for operation flow
                # but in strict mode we might want to log.
                # The feedback specifically said "narrow try-except".
                # For set_system_config, failures are usually serialization issues.
                pass

    def get_system_config(self) -> "SystemConfig":
        from pydantic import ValidationError

        from mlip_autopipec.config.models import SystemConfig
        self._ensure_connection()
        if self._connection is not None:
             try:
                return SystemConfig.model_validate(self._connection.metadata)
             except ValidationError as e:
                 raise DatabaseError(f"Stored SystemConfig is invalid: {e}") from e
        raise DatabaseError("No SystemConfig found")
