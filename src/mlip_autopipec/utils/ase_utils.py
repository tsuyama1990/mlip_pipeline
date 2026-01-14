from typing import Any

from ase import Atoms


def tag_atoms_with_metadata(atoms: Atoms, metadata: dict[str, Any]) -> Atoms:
    """
    Tags an ASE Atoms object with metadata in its info dictionary.
    """
    atoms.info.update(metadata)
    return atoms
