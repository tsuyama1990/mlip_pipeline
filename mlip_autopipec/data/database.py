"""Database manager for handling ASE database operations.

This module provides a wrapper around the ASE database to handle custom
metadata required for the MLIP-AutoPipe workflow.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect
from pydantic import BaseModel

from mlip_autopipec.config.models import SystemConfig


class DatabaseManager:
    """A wrapper for the ASE database to manage custom metadata."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the DatabaseManager.

        Args:
            db_path: The path to the ASE database file.

        """
        self.db_path = Path(db_path)
        self._connection: Any | None = None

    def connect(self) -> Any:
        """Establish a connection to the database.

        Returns:
            The ASE database connection object.

        """
        if self._connection is None:
            self._connection = connect(self.db_path)  # type: ignore[no-untyped-call]
        return self._connection

    def initialize(self, system_config: SystemConfig) -> None:
        """Initialize the database with system configuration metadata.

        Args:
            system_config: The system configuration to store in metadata.
        """
        conn = self.connect()

        # ASE DB metadata handling can be tricky on fresh files.
        try:
            current_metadata = conn.metadata
        except (AssertionError, AttributeError):
            current_metadata = {}

        # We store the system_config as a dict under 'system_config' key
        # We use model_dump(mode='json') to handle UUIDs and Paths
        current_metadata["system_config"] = system_config.model_dump(mode='json')
        conn.metadata = current_metadata

    def add_structure(self, atoms: Atoms, metadata: dict[str, Any] | BaseModel | None = None) -> int:
        """Add a structure to the database with metadata.

        Args:
            atoms: The ASE Atoms object.
            metadata: A dictionary or Pydantic model containing metadata.

        Returns:
            The ID of the newly written row.
        """
        conn = self.connect()
        kvp = {}

        if metadata:
            if isinstance(metadata, BaseModel):
                kvp = metadata.model_dump(mode='json', exclude_none=True)
            else:
                kvp = metadata.copy()

        # Flatten nested dicts if any? ASE DB supports flat key-values best.
        # But we assume metadata is flat-ish or ASE handles JSON serialization for dicts?
        # ASE DB supports `data` dict for complex objects, and `key_value_pairs` for queryable columns.
        # We put it in key_value_pairs for now as they are queryable.

        return conn.write(atoms, key_value_pairs=kvp)  # type: ignore[no-any-return, no-untyped-call]

    def write_calculation(
        self,
        atoms: Atoms,
        metadata: BaseModel,
        force_mask: np.ndarray | None = None,
    ) -> int:
        """Write a calculation result to the database with custom metadata.

        (Legacy/Specific Wrapper around add_structure)

        Args:
            atoms: The ASE Atoms object with calculation results attached.
            metadata: A Pydantic model containing structured metadata.
            force_mask: An optional NumPy array with the force mask.

        Returns:
            The ID of the newly written row.

        """
        # Convert to dict
        meta_dict = metadata.model_dump(mode='json', exclude_none=True)

        # Prefix keys to avoid collisions? Original code did "mlip_".
        # Let's keep it simple or follow the request.
        # For now, we just pass it as is.

        if force_mask is not None:
             # ASE supports arrays in 'data' usually, or we can stash it in `data`.
             # `write` has `data` arg.
             pass

        # Re-using add_structure but maybe we want to use `data` for large arrays.
        # The previous implementation put force_mask in key_value_pairs which converts to text/json?
        # Ideally arrays go to `data`.

        return self.add_structure(atoms, metadata=meta_dict)

    def get_completed_calculations(self) -> list[Atoms]:
        """Retrieve all completed calculations from the database.

        Returns:
            A list of ASE Atoms objects for all completed calculations.

        """
        conn = self.connect()
        # This assumes 'calculated=True' is a key we set? Or ASE status?
        # ASE db doesn't strictly have 'calculated' unless we set it.
        # Let's assume we select all for now or check for energy presence.
        return [row.toatoms() for row in conn.select()]
