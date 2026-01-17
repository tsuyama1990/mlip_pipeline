import contextlib
from pathlib import Path
from typing import Any
from typing import Any as AtomsObject

import ase.db
import numpy as np
from ase.db.core import Database
from pydantic import ValidationError

from mlip_autopipec.config.models import DFTResult, SystemConfig, TrainingData
from mlip_autopipec.exceptions import DatabaseError


class DatabaseManager:
    """
    Wrapper around ase.db to enforce schema and metadata requirements.
    Acts as the single source of truth for all database interactions.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._connection: Database | None = None

    def initialize(self) -> None:
        """
        Initializes the database connection.
        If the database does not exist, it is created.
        """
        try:
            # Ensure parent directory exists (though WorkspaceManager handles this, redundancy is safe)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect (or create)
            self._connection = ase.db.connect(str(self.db_path))

            # Secure the database file
            # Might fail on Windows or some filesystems, log warning but don't crash
            # Since logger is not imported here to keep it simple, we just pass
            # or could import logging if strictly required.
            with contextlib.suppress(OSError):
                self.db_path.chmod(0o600)

            # Force initialization
            self._connection.count()
        except Exception as e:
            msg = f"Failed to initialize database at {self.db_path}: {e}"
            raise DatabaseError(msg) from e

    def set_system_config(self, config: SystemConfig) -> None:
        """
        Stores the system configuration in the database metadata.
        This must be called during initialization or setup.
        """
        if self._connection is None:
            self.initialize()

        try:
            config_dict = config.model_dump(mode="json")

            # Update metadata
            if self._connection is None:
                msg = "Database connection is None after initialization."
                raise DatabaseError(msg)

            current_metadata = self._connection.metadata.copy()
            current_metadata.update(config_dict)
            self._connection.metadata = current_metadata
        except Exception as e:
            msg = "Failed to write system configuration to database metadata."
            raise DatabaseError(msg) from e

    def get_metadata(self) -> dict[str, Any]:
        """
        Retrieves the metadata from the database.
        """
        if self._connection is None:
            self.initialize()

        if self._connection is None:
            msg = "Database connection is None after initialization."
            raise DatabaseError(msg)

        return self._connection.metadata  # type: ignore

    def get_system_config(self) -> SystemConfig:
        """
        Retrieves and validates the SystemConfig stored in the database metadata.

        Returns:
            A validated SystemConfig object.

        Raises:
            DatabaseError: If metadata is missing or invalid.
        """
        metadata = self.get_metadata()
        try:
            # We assume the config is stored at the top level or under specific keys.
            # set_system_config dumps the whole model into metadata.
            return SystemConfig.model_validate(metadata)
        except ValidationError as e:
            msg = f"Database metadata does not contain a valid SystemConfig: {e}"
            raise DatabaseError(msg) from e

    def get_atoms(self, selection: str | None = None) -> list[AtomsObject]:
        """
        Retrieves atoms objects from the database.

        Args:
            selection: ASE DB selection string (e.g. 'energy<0').

        Returns:
            List of ASE Atoms objects.
        """
        if self._connection is None:
            self.initialize()

        if self._connection is None:
            msg = "Database connection is None."
            raise DatabaseError(msg)

        try:
            # ase.db.select returns an iterator
            rows = self._connection.select(selection=selection)
            return [row.toatoms() for row in rows]
        except Exception as e:
            msg = f"Failed to retrieve atoms from database: {e}"
            raise DatabaseError(msg) from e

    def get_training_data(self) -> list[AtomsObject]:
        """
        Retrieves atoms specifically formatted and validated for training.
        Use this instead of raw get_atoms when preparing for Pacemaker.

        Returns:
             List of ASE Atoms objects with info['energy'] and arrays['forces'] populated.
        """
        if self._connection is None:
            self.initialize()

        if self._connection is None:
            msg = "Database connection is None."
            raise DatabaseError(msg)

        atoms_list = []
        try:
            for row in self._connection.select():
                # Validate that the row has the required data fields
                # We use the TrainingData Pydantic model for validation
                try:
                    # row.data contains the dictionary of key-value pairs stored with the atoms
                    validated_data = TrainingData(**row.data)

                    atoms = row.toatoms()
                    # Map validated data to atoms attributes expected by Pacemaker/ASE
                    atoms.info["energy"] = validated_data.energy
                    atoms.arrays["forces"] = np.array(validated_data.forces)
                    atoms_list.append(atoms)
                except ValidationError:
                    # Skip rows that don't match training data schema (maybe failed jobs or different types)
                    # Alternatively, we could log a warning.
                    continue
        except Exception as e:
            msg = f"Failed to read training data: {e}"
            raise DatabaseError(msg) from e

        return atoms_list

    def save_dft_result(
        self, atoms: AtomsObject, result: DFTResult, metadata: dict[str, Any]
    ) -> None:
        """
        Saves a DFT calculation result and its metadata.
        """
        if self._connection is None:
            self.initialize()

        if self._connection is None:
            msg = "Database connection is None."
            raise DatabaseError(msg)

        try:
            # Separate metadata for info dict and arrays
            info_metadata = metadata.copy()
            force_mask = info_metadata.pop("force_mask", None)

            atoms.info["energy"] = result.energy
            atoms.info["forces"] = result.forces
            atoms.info["stress"] = result.stress
            atoms.info.update(info_metadata)

            if force_mask is not None:
                atoms.arrays["force_mask"] = np.array(force_mask)

            # Check if exists logic could go here, but for now we write
            self._connection.write(atoms, data=result.model_dump())
            # Note: We save result.model_dump() into 'data' so we can reconstruct TrainingData later
        except Exception as e:
            msg = f"Failed to save DFT result: {e}"
            raise DatabaseError(msg) from e
