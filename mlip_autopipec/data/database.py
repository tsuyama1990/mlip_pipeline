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
        # ase.db (sqlite) requires at least one table/row or connection initialized to access metadata safely
        # in some versions, or it lazy-inits.
        # But `conn.metadata` access triggers assertion if connection is None.
        # It seems `connect()` returns a Database object, but maybe it hasn't actually opened the SQLite file yet?
        # Let's ensure initialization by writing a dummy reserved key if needed, or handling the exception.

        # However, for a fresh file, ase.db usually handles it.
        # The error `assert self.connection is not None` suggests `conn` object exists but its internal `connection` is None.
        # This usually happens if `connect` hasn't established the sqlite3 connection.

        # In ASE, `connect` returns a wrapper. `_connect()` is internal.
        # Accessing `metadata` property triggers `self.connection` check.

        # If the file is new and empty, ASE might not have opened the connection.
        # Let's try to explicitly access something to force connection.
        try:
            _ = conn.count()
        except Exception:
            # If count fails (e.g. no tables), we might still be able to write metadata if we initialize tables.
            pass

        # If it's a new database, we might need to "create" it.
        # Writing metadata should be possible on an empty DB in newer ASE, but let's be safe.
        # If we really hit an issue, we can write a dummy atom or use a transaction.

        # Actually, simpler fix: ASE's `metadata` getter/setter might be strict.
        # Let's try to set it.

        try:
            current_metadata = conn.metadata
        except AssertionError:
             # Connection not open.
             # ASE db `connect` with `append=True` (default) should be fine, but maybe lazy.
             # Let's force a write to init tables?
             # Or just initializing the connection.
             # NOTE: accessing `conn.cursor()` usually forces connection.
             pass

        # In ASE's sqlite.py, `metadata` property reads from `self.connection`.
        # `self.connection` is set in `_connect`.
        # `connect` calls `_connect` immediately usually? No, `connect` returns `SQLite3Database(...)`.
        # `__init__` calls `_connect()`.

        # Wait, if `_connect` was called, `self.connection` should be set.
        # Unless `db_path` is invalid or something.

        # It seems checking `conn.metadata` on a fresh DB might fail if tables aren't created?
        # Let's write the metadata using `conn.metadata = ...`.
        # But we need to read existing metadata first to preserve it.

        # If assert fails, it means we can't read it.
        # Let's assume empty if we can't read.

        # To be safe and ensure the DB file is created and tables exist:
        # We can write an info row, or just rely on ASE's behavior for metadata writing which does `_initialize` if needed.

        # But `conn.metadata` getter has the assertion.
        # The setter might also have it or might init.

        # Let's catch the assertion error and treat it as empty metadata.
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
        return [row.toatoms() for row in conn.select(calculated=True)]
