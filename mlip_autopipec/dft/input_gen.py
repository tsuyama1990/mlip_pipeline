"""
Input generation for Quantum Espresso.
"""
from pathlib import Path
from typing import Any, Dict, Union

from ase import Atoms
from ase.io import write


def write_pw_input(
    atoms: Atoms,
    parameters: Dict[str, Any],
    pseudopotentials: Dict[str, str],
    kpts: Any,
    output_path: Union[str, Path],
) -> None:
    """
    Writes the Quantum Espresso input file.

    Args:
        atoms: The atomic structure.
        parameters: Dictionary of calculation parameters (namelists).
        pseudopotentials: Dictionary mapping elements to UPF filenames.
        kpts: K-points grid (e.g. [4, 4, 4]) or None.
        output_path: Path to write the input file.
    """
    # Helper to merge defaults
    input_data = parameters.copy()

    # Ensure CONTROL namelist exists
    if "control" not in input_data:
        input_data["control"] = {}

    # Enforce mandatory flags
    # Spec: tprnfor=True, tstress=True, disk_io='low'
    defaults = {
        "tprnfor": True,
        "tstress": True,
        "disk_io": "low",
    }

    for key, val in defaults.items():
        # ASE uses lowercase keys for namelist params usually, but checks both.
        # We enforce them.
        input_data["control"][key] = val

    write(
        output_path,
        atoms,
        format="espresso-in",
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts,
    )
