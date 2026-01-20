import io
from typing import Any

from ase import Atoms
from ase.io.espresso import write_espresso_in


def write_pw_input(
    atoms: Atoms,
    parameters: dict[str, Any],
    pseudopotentials: dict[str, str],
    kpts: tuple[int, int, int]
) -> str:
    """
    Generates QE input string.
    """
    input_data = parameters.copy()

    # Ensure control flags
    if "control" not in input_data:
        input_data["control"] = {}

    # Ensure calculation type is set (default to scf if not provided)
    # Spec says "Check that calculation='scf' is present" in tests
    if "calculation" not in input_data["control"]:
        input_data["control"]["calculation"] = "scf"

    input_data["control"]["tprnfor"] = True
    input_data["control"]["tstress"] = True
    input_data["control"]["disk_io"] = "low"

    # Ensure system flags
    if "system" not in input_data:
        input_data["system"] = {}

    # Use ASE to write to string
    fd = io.StringIO()
    write_espresso_in(
        fd,
        atoms,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts
    )
    return fd.getvalue()
