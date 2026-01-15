"""Database manager for handling ASE database operations.

This module provides a wrapper around the ASE database to handle custom
metadata required for the MLIP-AutoPipe workflow.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.db.row import AtomsRow

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
            self._connection = connect(self.db_path)
        return self._connection

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
            force_mask: An optional numpy array defining the force mask.

        Returns:
            The ID of the newly written row.

        """
        conn = self.connect()
        key_value_pairs = {f"mlip_{k}": v for k, v in metadata.model_dump().items()}
        if force_mask is not None:
            key_value_pairs["force_mask"] = force_mask
        return conn.write(atoms, key_value_pairs=key_value_pairs)

    def get_calculations_for_training(self) -> list[Atoms]:
        """Retrieve all completed calculations for training.

        This method retrieves the Atoms object and attaches the force_mask to the
        `atoms.info` dictionary if it exists in the database.

        Returns:
            A list of ASE Atoms objects ready for the trainer.

        """
        conn = self.connect()
        atoms_list = []
        for row in conn.select(calculated=True):
            atoms = row.toatoms()
            if "force_mask" in row.key_value_pairs:
                atoms.info["force_mask"] = row.key_value_pairs["force_mask"]
            atoms_list.append(atoms)
        return atoms_list
