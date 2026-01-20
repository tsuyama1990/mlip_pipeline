import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing import Any as AtomsObject

import ase.db
import numpy as np
from ase.db.core import Database
from pydantic import ValidationError

from mlip_autopipec.exceptions import DatabaseException

if TYPE_CHECKING:
    from mlip_autopipec.config.schemas.system import SystemConfig
    from mlip_autopipec.data_models.dft_models import DFTResult


class _DatabaseReader:
    """
    Handles read operations for the database.
    """

    def __init__(self, connection: Database | None):
        self._connection = connection

    def _ensure_connection(self) -> Database:
        if self._connection is None:
            msg = "Database connection is lost."
            raise DatabaseException(msg)
        return self._connection

    def get_metadata(self) -> dict[str, Any]:
        conn = self._ensure_connection()
        return conn.metadata  # type: ignore

    def get_atoms(self, selection: str | None = None) -> list[AtomsObject]:
        conn = self._ensure_connection()
        try:
            rows = conn.select(selection=selection)
            return [row.toatoms() for row in rows]
        except Exception as e:
            msg = f"Failed to retrieve atoms from database: {e}"
            raise DatabaseException(msg) from e

    def count(self, selection: str | None = None) -> int:
        conn = self._ensure_connection()
        try:
            return int(conn.count(selection=selection))
        except Exception as e:
            msg = f"Failed to count rows: {e}"
            raise DatabaseException(msg) from e


class _DatabaseWriter:
    """
    Handles write operations for the database.
    """

    def __init__(self, connection: Database | None):
        self._connection = connection

    def _ensure_connection(self) -> Database:
        if self._connection is None:
            msg = "Database connection is lost."
            raise DatabaseException(msg)
        return self._connection

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        conn = self._ensure_connection()
        try:
            current_metadata = conn.metadata.copy()
            current_metadata.update(metadata)
            conn.metadata = current_metadata
        except Exception as e:
            msg = "Failed to update database metadata."
            raise DatabaseException(msg) from e

    def write_atoms(self, atoms: AtomsObject, key_value_pairs: dict[str, Any] | None = None, data: dict[str, Any] | None = None) -> int:
        conn = self._ensure_connection()
        try:
            return conn.write(atoms, key_value_pairs=key_value_pairs, data=data)
        except Exception as e:
            msg = f"Failed to write atoms: {e}"
            raise DatabaseException(msg) from e


class DatabaseManager:
    """
    Wrapper around ase.db to enforce schema and metadata requirements.
    Acts as the facade for all database interactions, delegating to Reader/Writer.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._connection: Database | None = None
        self._reader: _DatabaseReader | None = None
        self._writer: _DatabaseWriter | None = None

    def initialize(self) -> None:
        """
        Initializes the database connection.
        If the database does not exist, it is created.
        """
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect (or create)
            self._connection = ase.db.connect(str(self.db_path))

            # Secure the database file
            with contextlib.suppress(OSError):
                self.db_path.chmod(0o600)

            # Force initialization
            self._connection.count()

            # Initialize components
            self._reader = _DatabaseReader(self._connection)
            self._writer = _DatabaseWriter(self._connection)

        except Exception as e:
            msg = f"Failed to initialize database at {self.db_path}: {e}"
            raise DatabaseException(msg) from e

    def _ensure_initialized(self) -> None:
        if self._connection is None or self._reader is None or self._writer is None:
            self.initialize()
            if self._connection is None:
                msg = "Database connection is None after initialization."
                raise DatabaseException(msg)

    # --- Configuration Methods ---

    def set_system_config(self, config: "SystemConfig") -> None:
        self._ensure_initialized()
        assert self._writer is not None # for mypy
        config_dict = config.model_dump(mode="json")
        self._writer.update_metadata(config_dict)

    def get_metadata(self) -> dict[str, Any]:
        self._ensure_initialized()
        assert self._reader is not None
        return self._reader.get_metadata()

    def get_system_config(self) -> "SystemConfig":
        from mlip_autopipec.config.schemas.system import SystemConfig

        metadata = self.get_metadata()
        try:
            return SystemConfig.model_validate(metadata)
        except ValidationError as e:
            msg = f"Database metadata does not contain a valid SystemConfig: {e}"
            raise DatabaseException(msg) from e

    # --- Data Access Methods ---

    def get_atoms(self, selection: str | None = None) -> list[AtomsObject]:
        self._ensure_initialized()
        assert self._reader is not None
        return self._reader.get_atoms(selection)

    def count(self, selection: str | None = None) -> int:
        self._ensure_initialized()
        assert self._reader is not None
        return self._reader.count(selection)

    def get_training_data(self) -> list[AtomsObject]:
        from mlip_autopipec.config.schemas.training import TrainingData

        self._ensure_initialized()
        # Direct access needed for iteration, or expose iterator in Reader
        # For simplicity, implementing here using underlying connection or adding method to Reader
        # Let's add specialized method to Reader? Or keep high level logic here?
        # Keeping high level logic here but using reader/writer primitives is safer?
        # Actually, get_training_data involves validation logic.

        # We can use the connection directly or use select from Reader if we modify Reader.
        # Let's trust _ensure_initialized.
        assert self._connection is not None

        atoms_list = []
        try:
            for row in self._connection.select():
                try:
                    validated_data = TrainingData(**row.data)
                    atoms = row.toatoms()
                    atoms.info["energy"] = validated_data.energy
                    atoms.arrays["forces"] = np.array(validated_data.forces)
                    atoms_list.append(atoms)
                except ValidationError:
                    continue
        except Exception as e:
            msg = f"Failed to read training data: {e}"
            raise DatabaseException(msg) from e

        return atoms_list

    # --- Write Methods ---

    def save_dft_result(
        self, atoms: AtomsObject, result: "DFTResult", metadata: dict[str, Any]
    ) -> None:
        self._ensure_initialized()
        assert self._writer is not None

        info_metadata = metadata.copy()
        force_mask = info_metadata.pop("force_mask", None)

        atoms.info["energy"] = result.energy
        atoms.info["forces"] = result.forces
        atoms.info["stress"] = result.stress
        atoms.info.update(info_metadata)

        if force_mask is not None:
            atoms.arrays["force_mask"] = np.array(force_mask)

        self._writer.write_atoms(atoms, data=result.model_dump())

    def save_candidate(self, atoms: AtomsObject, metadata: dict[str, Any]) -> None:
        from mlip_autopipec.data_models.candidate import CandidateData

        self._ensure_initialized()
        assert self._writer is not None

        try:
            CandidateData(**metadata)
            atoms.info.update(metadata)
            # Use key_value_pairs for searchable metadata (like status)
            self._writer.write_atoms(atoms, key_value_pairs=metadata)
        except ValidationError as e:
            msg = f"Invalid candidate metadata: {e}"
            raise DatabaseException(msg) from e

    def add_calculation(self, atoms: AtomsObject, metadata: dict[str, Any]) -> int:
        self._ensure_initialized()
        assert self._writer is not None
        # metadata usually goes to key_value_pairs for search
        return self._writer.write_atoms(atoms, key_value_pairs=metadata)

    def get_pending_calculations(self) -> list[AtomsObject]:
        return self.get_atoms(selection="status=pending")
