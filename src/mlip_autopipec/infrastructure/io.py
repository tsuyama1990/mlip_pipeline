from pathlib import Path
from typing import Any

import ase.io
import yaml

from mlip_autopipec.domain_models.structure import Structure


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Safely load a YAML file.

    WARNING: This function loads the entire file into memory.
    It is intended for configuration files, not large datasets.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is invalid.
        TypeError: If the YAML content is not a dictionary.
    """
    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        msg = f"YAML file {path} must contain a dictionary, got {type(data)}"
        raise TypeError(msg)

    return data


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    """
    Dump data to a YAML file.

    Args:
        data: Dictionary to dump.
        path: Path to the output file.
    """
    with path.open("w") as f:
        yaml.dump(data, f, sort_keys=False)


def load_structures(path: Path) -> list[Structure]:
    """
    Load structures from a file (xyz, extxyz, poscar, etc.).

    Args:
        path: Path to the structure file.

    Returns:
        List of Structure objects.
    """
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")

    # ase.io.read returns Atoms or list of Atoms
    # index=':' reads all
    try:
        atoms_list = ase.io.read(path, index=":")
    except Exception as e:
        msg = f"Failed to read structures from {path}: {e}"
        raise RuntimeError(msg) from e

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list] # type: ignore

    return [Structure.from_ase(atoms) for atoms in atoms_list]
