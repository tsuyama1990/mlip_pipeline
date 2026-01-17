"""Database manager for handling ASE database operations.

This module provides a wrapper around the ASE database to handle custom
metadata required for the MLIP-AutoPipe workflow.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect

from mlip_autopipec.config.models import CalculationMetadata, SystemConfig


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

        # Force connection initialization if it hasn't happened yet.
        # This handles cases where ASE delays opening the SQLite file.
        try:
            conn.count()
        except Exception:
            pass

        try:
            current_metadata = conn.metadata
        except (AssertionError, AttributeError):
            current_metadata = {}

        # We store the system_config as a dict under 'system_config' key
        # We use model_dump(mode='json') to handle UUIDs and Paths
        current_metadata["system_config"] = system_config.model_dump(mode='json')
        conn.metadata = current_metadata

    def write_calculation(
        self,
        atoms: Atoms,
        metadata: CalculationMetadata,
        force_mask: np.ndarray | None = None,
    ) -> int:
        """Write a calculation result to the database with custom metadata.

        Args:
            atoms: The ASE Atoms object with calculation results attached.
            metadata: A `CalculationMetadata` object containing structured metadata.
            force_mask: An optional NumPy array with the force mask.

        Returns:
            The ID of the newly written row.

        """
        conn = self.connect()
        kvp = {f"mlip_{k}": v for k, v in metadata.model_dump().items() if v is not None}
        if force_mask is not None:
            kvp["mlip_force_mask"] = force_mask.tolist()

        return conn.write(atoms, key_value_pairs=kvp)  # type: ignore[no-any-return]

    def get_completed_calculations(self) -> list[Atoms]:
        """Retrieve all completed calculations from the database.

        Returns:
            A list of ASE Atoms objects for all completed calculations.

        """
        conn = self.connect()
        # Note: calculated=True filters for rows with valid energy/forces
        # ASE's calculated=True can be strict about calculator checks.
        # We manually filter to be safe.
        return [
            row.toatoms()
            for row in conn.select()
            if hasattr(row, "energy") and row.energy is not None
        ]
