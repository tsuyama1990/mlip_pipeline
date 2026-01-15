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

# Note: Using `Any` for ase.Atoms is a pragmatic choice, consistent with
# the Pydantic models, as it is a complex object.
from typing import Any as AtomsObject

from ase.db import connect
from ase.db.row import AtomsRow

from mlip_autopipec.config.models import DFTResult


def save_dft_result(
    db_path: Path,
    atoms: AtomsObject,
    result: DFTResult,
    config_type: str,
) -> None:
    """
    Saves a DFT calculation result to the specified ASE database.

    This function connects to the database, attaches the energy, forces, and
    stress from the `DFTResult` object to the `ase.Atoms` object's `info`
    dictionary, and then writes the a toms object to the database.

    Args:
        db_path: The path to the ASE database file (e.g., an SQLite file).
        atoms: The `ase.Atoms` object that was calculated.
        result: The `DFTResult` object containing the calculation output.
        config_type: A label to categorize the structure (e.g.,
                     'initial_training_set').
    """
    with connect(db_path) as db:
        # Attach results and metadata to the Atoms object before saving
        atoms.info["energy"] = result.energy
        atoms.info["forces"] = result.forces
        atoms.info["stress"] = result.stress
        atoms.info["job_id"] = str(result.job_id)
        atoms.info["config_type"] = config_type
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
