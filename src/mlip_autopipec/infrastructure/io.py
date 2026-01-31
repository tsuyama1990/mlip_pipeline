from pathlib import Path
from typing import Any, Iterator

import yaml
from ase.io import iread

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


def load_structures(path: Path) -> Iterator[Structure]:
    """
    Load structures from a file (xyz, extxyz, poscar, etc.).
    Returns an iterator to be memory efficient.

    Args:
        path: Path to the structure file.

    Returns:
        Iterator of Structure objects.
    """
    # type: ignore[no-untyped-call]
    for atoms in iread(path, index=":"):
        yield Structure.from_ase(atoms)
