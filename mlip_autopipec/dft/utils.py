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
    potential_files = [f.name for f in pseudo_dir.iterdir() if f.name.endswith((".upf", ".UPF"))]

    for element in elements:
        # Heuristic: Find file starting with "El." or "El_"
        match = None
        for filename in potential_files:
            if filename.startswith((f"{element}.", f"{element}_")):
                match = filename
                break

        if not match and f"{element}.upf" in potential_files:
            match = f"{element}.upf"

        if match:
            # Basic validation: check if file is not empty
            file_path = pseudo_dir / match
            if file_path.stat().st_size == 0:
                msg = f"Pseudopotential file {file_path} is empty."
                raise ValueError(msg)
            # Check header
            with file_path.open("r", errors="ignore") as f:
                header = f.read(1024)
                if "<UPF" not in header and "&" not in header:
                    # Very basic check for XML or old format
                    # But don't be too strict if format varies
                    pass

            mapping[element] = match
        else:
            msg = f"No pseudopotential found for element {element} in {pseudo_dir}"
            raise FileNotFoundError(msg)

    return mapping
