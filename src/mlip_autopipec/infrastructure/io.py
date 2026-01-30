import shlex
import subprocess
from pathlib import Path
from typing import Any, cast

import ase.io
import ase
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


def write_lammps_data(structure: Structure, path: Path) -> None:
    """
    Write structure to LAMMPS data file.

    Args:
        structure: The Structure object to write.
        path: Output file path.
    """
    atoms = structure.to_ase()
    ase.io.write(path, atoms, format="lammps-data")  # type: ignore[no-untyped-call]


def read_lammps_dump(path: Path, species: list[str] | None = None) -> Structure:
    """
    Read the last frame from a LAMMPS dump file.

    Args:
        path: Path to dump file.
        species: List of chemical symbols to map atom types to (e.g. ["Si", "O"]).
                 If provided, type 1 becomes species[0], etc.

    Returns:
        The parsed Structure.
    """
    # Use index=-1 to get the last frame
    atoms_obj = ase.io.read(path, format="lammps-dump-text", index=-1)  # type: ignore[no-untyped-call]
    atoms = cast(ase.Atoms, atoms_obj)

    if species:
        # Assuming atom numbers/types in the dump correspond to 1-based indices in species list
        current_numbers = atoms.get_atomic_numbers()  # type: ignore[no-untyped-call]
        new_symbols = [species[n - 1] for n in current_numbers]
        atoms.set_chemical_symbols(new_symbols)  # type: ignore[no-untyped-call]

    return Structure.from_ase(atoms)


def run_subprocess(
    command: str, timeout: float = 3600.0, cwd: Path | None = None
) -> None:
    """
    Run a shell command with timeout.

    Args:
        command: The command string.
        timeout: Max duration in seconds.
        cwd: Working directory.

    Raises:
        RuntimeError: If command fails or times out.
    """
    args = shlex.split(command)
    try:
        subprocess.run(
            args,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        msg = f"Command timed out after {timeout}s: {command}"
        raise RuntimeError(msg) from e
    except subprocess.CalledProcessError as e:
        msg = f"Command failed with code {e.returncode}: {e.stderr}"
        raise RuntimeError(msg) from e
