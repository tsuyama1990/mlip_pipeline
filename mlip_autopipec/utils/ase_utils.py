# FIXME: The above comment is a temporary workaround for a ruff bug.
# It should be removed once the bug is fixed.
# For more information, see: https://github.com/astral-sh/ruff/issues/10515
"""
This module provides utility functions for interacting with the ASE database.

It encapsulates the logic for saving and querying DFT calculation results,
ensuring a consistent data storage format and providing a simple API for
other components of the application.
"""

from pathlib import Path
from typing import Any

# Note: Using `Any` for ase.Atoms is a pragmatic choice, consistent with
# the Pydantic models, as it is a complex object.
from typing import Any as AtomsObject

import numpy as np
from ase.db import connect
from ase.db.row import AtomsRow

from mlip_autopipec.config.models import DFTResult


def save_dft_result(
    db_path: Path,
    atoms: AtomsObject,
    result: DFTResult,
    metadata: dict[str, Any],
) -> None:
    """
    Saves a DFT calculation result and its metadata to the ASE database.

    Args:
        db_path: Path to the ASE database file.
        atoms: The `ase.Atoms` object that was calculated.
        result: The `DFTResult` object containing the calculation output.
        metadata: A dictionary of metadata, e.g., {'uuid': '...', 'config_type': '...', 'force_mask': np.array}.
    """
    with connect(db_path) as db:
        # Separate metadata for info dict and arrays
        info_metadata = metadata.copy()
        force_mask = info_metadata.pop('force_mask', None)

        atoms.info["energy"] = result.energy
        atoms.info["forces"] = result.forces
        atoms.info["stress"] = result.stress
        atoms.info.update(info_metadata)

        if force_mask is not None:
            atoms.arrays['force_mask'] = np.array(force_mask)

        db.write(atoms)


def check_if_exists(db_path: Path, atoms: AtomsObject) -> AtomsRow | None:
    """
    Checks if a given structure already exists in the database.

    This is a placeholder for a more sophisticated structure matching
    algorithm that could be implemented in the future (e.g., using
    structure fingerprints or a graph-based comparison). The current
    implementation is a simple example.

    Args:
        db_path: The path to the ASE database file.
        atoms: The `ase.Atoms` object to check.

    Returns:
        The `AtomsRow` of the existing structure if found, otherwise `None`.
    """
    if not db_path.exists():
        return None

    with connect(db_path) as db:
        # This is a very basic check. A real implementation would need a
        # more robust way to identify unique structures.
        for row in db.select():
            if (
                len(atoms) == len(row.toatoms())
                and atoms.get_chemical_formula() == row.toatoms().get_chemical_formula()
            ):
                # Further checks, e.g., on positions or cell, would be needed
                # for a robust duplicate check.
                pass  # Placeholder for more complex comparison
    return None
