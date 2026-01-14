"""Database manager for handling ASE database operations.

This module provides a wrapper around the ASE database to handle custom
metadata required for the MLIP-AutoPipe workflow.
"""

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.db import connect

from mlip_autopipec.config_schemas import CalculationMetadata


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

    def write_calculation(self, atoms: Atoms, metadata: CalculationMetadata) -> int:
        """Write a calculation result to the database with custom metadata.

        Args:
            atoms: The ASE Atoms object with calculation results attached.
            metadata: A `CalculationMetadata` object containing structured metadata.

        Returns:
            The ID of the newly written row.

        """
        conn = self.connect()
        prefixed_metadata = {f"mlip_{k}": v for k, v in metadata.model_dump().items()}
        return conn.write(atoms, key_value_pairs=prefixed_metadata)  # type: ignore[no-any-return]

    def get_completed_calculations(self) -> list[Atoms]:
        """Retrieve all completed calculations from the database.

        Returns:
            A list of ASE Atoms objects for all completed calculations.

        """
        conn = self.connect()
        return [row.toatoms() for row in conn.select(calculated=True)]
