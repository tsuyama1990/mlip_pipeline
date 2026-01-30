import subprocess
from pathlib import Path
from typing import Any, Optional

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
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None
) -> tuple[str, str]:
    """
    Run a subprocess command.

    Args:
        command: List of command arguments.
        timeout: Timeout in seconds.
        cwd: Working directory.
        env: Environment variables.

    Returns:
        Tuple of (stdout, stderr).

    Raises:
        subprocess.CalledProcessError: If the command fails.
        TimeoutError: If the command times out.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=True
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Command timed out after {timeout}s: {command}") from e


def write_lammps_data(structure: Structure, path: Path) -> None:
    """
    Write structure to LAMMPS data file.

    Args:
        structure: Structure to write.
        path: Path to the output file.
    """
    atoms = structure.to_ase()
    # ase.io.write supports lammps-data
    ase.io.write(path, atoms, format="lammps-data")  # type: ignore[no-untyped-call]


def read_lammps_dump(path: Path) -> Structure:
    """
    Read structure from LAMMPS dump file (last frame).

    Args:
        path: Path to the dump file.

    Returns:
        Structure object.
    """
    # index=-1 gets the last frame
    atoms = ase.io.read(path, index=-1, format="lammps-dump-text")  # type: ignore[no-untyped-call]

    # ASE read returns Atoms or list of Atoms depending on index.
    # With index=-1, it returns a single Atoms object.
    if isinstance(atoms, list):
         # Should not happen with index=-1 but safe guard
         atoms = atoms[-1]

    return Structure.from_ase(atoms)
