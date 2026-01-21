from pathlib import Path
from typing import Any

import ase.io
from ase import Atoms


def write_pw_input(
    atoms: Atoms,
    filename: Path,
    input_data: dict[str, Any],
    pseudopotentials: dict[str, str],
    kpts: list | float | int | None = None,
    kspacing: float | None = None,
    koffset: list | int | None = None,
) -> None:
    """
    Writes a Quantum Espresso input file using ASE.
    Augments the input_data with required flags (tprnfor, tstress, disk_io).
    """
    # Create a copy to avoid modifying the original dict
    params = input_data.copy()

    # Ensure sections exist
    if "control" not in params:
        params["control"] = {}
    if "system" not in params:
        params["system"] = {}

    # Enforce required flags for force/stress calculation and IO
    params["control"]["disk_io"] = "low"
    params["control"]["tprnfor"] = True
    params["control"]["tstress"] = True

    # Ensure calculation type is set (default scf)
    if "calculation" not in params["control"]:
        params["control"]["calculation"] = "scf"

    # Default koffset to (0, 0, 0) if not provided, to prevent ASE error
    if koffset is None:
        koffset = (0, 0, 0)

    with filename.open("w") as f:
        ase.io.write(
            f,
            atoms,
            format="espresso-in",
            input_data=params,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
            kspacing=kspacing,
            koffset=koffset,
        )
