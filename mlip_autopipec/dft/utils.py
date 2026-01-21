from pathlib import Path

import numpy as np
from ase import Atoms

MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Mn", "Cr"}

def get_kpoints(atoms: Atoms, density: float) -> list[int]:
    """
    Calculates the Monkhorst-Pack grid based on reciprocal lattice vectors.
    k_i = max(1, round(|b_i| * density))

    Args:
        atoms: The ASE Atoms object.
        density: K-point density (linear density).

    Returns:
        List[int]: The K-point grid [k1, k2, k3].
    """
    # Calculate reciprocal lattice vectors (with 2pi factor)
    # ase.cell.reciprocal() returns vectors without 2pi factor if documentation is confusing,
    # but let's check standard ASE behavior:
    # cell.reciprocal() gives vectors b_i such that a_i . b_j = delta_ij.
    # So |b_i| = 1/|a_i| (for cubic).
    # Physics definition is 2pi/|a_i|.
    # Memory says "multiplies ASE's `cell.reciprocal()` norms by 2π".
    recip = atoms.cell.reciprocal() * 2 * np.pi
    norms = np.linalg.norm(recip, axis=1)

    kpoints = []
    for norm in norms:
        k = max(1, round(norm * density))
        kpoints.append(k)

    return kpoints

def is_magnetic(atoms: Atoms) -> bool:
    """Checks if the structure contains magnetic elements."""
    return any(atom.symbol in MAGNETIC_ELEMENTS for atom in atoms)

def get_sssp_pseudopotentials(elements: list[str], pseudo_dir: Path) -> dict[str, str]:
    """
    Maps elements to UPF filenames in the pseudopotential directory.

    Args:
        elements: List of chemical symbols (e.g. ['Al', 'Fe']).
        pseudo_dir: Directory containing pseudopotential files.

    Returns:
        Dictionary mapping element symbol to filename.

    Raises:
        FileNotFoundError: If a pseudopotential for an element is not found.
    """
    mapping = {}
    if not pseudo_dir.exists():
        msg = f"Pseudopotential directory not found: {pseudo_dir}"
        raise FileNotFoundError(msg)

    # List all UPF files
    files = [f.name for f in pseudo_dir.iterdir() if f.name.endswith((".upf", ".UPF"))]

    for el in elements:
        # Heuristic: Find file starting with "El." or "El_"
        # Also need to match exact element (e.g. C vs Ca)
        # Regex or simple check
        match = None
        for f in files:
            # Check prefix
            # Valid separators: . or _
            if f.startswith((f"{el}.", f"{el}_")):
                match = f
                break

        if not match and f"{el}.upf" in files:
            match = f"{el}.upf"

        if match:
            mapping[el] = match
        else:
            msg = f"No pseudopotential found for element {el} in {pseudo_dir}"
            raise FileNotFoundError(msg)

    return mapping
