import subprocess
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


def run_subprocess(
    command: list[str],
    cwd: Path,
    timeout: float | None = None,
    check: bool = True,
) -> tuple[str, str]:
    """
    Run a subprocess command.

    Args:
        command: Command and arguments list.
        cwd: Working directory.
        timeout: Timeout in seconds.
        check: Whether to raise CalledProcessError on non-zero exit.

    Returns:
        Tuple of (stdout, stderr).
    """
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )
    return result.stdout, result.stderr


def write_lammps_data(structure: Structure, path: Path) -> None:
    """
    Write structure to LAMMPS data file.

    Args:
        structure: The structure to write.
        path: Output file path.
    """
    atoms = structure.to_ase()
    # Use atom_style='atomic' for basic simulations
    ase.io.write(path, atoms, format="lammps-data", atom_style="atomic")  # type: ignore[no-untyped-call]


def read_lammps_dump(path: Path) -> list[ase.Atoms]:
    """
    Read LAMMPS dump file.

    Args:
        path: Path to the dump file.

    Returns:
        List of ASE Atoms objects (trajectory).
    """
    return ase.io.read(path, format="lammps-dump-text", index=":")  # type: ignore[no-untyped-call, return-value]
