"""
Utility functions for DFT calculations.
"""

import numpy as np
from ase import Atoms


def get_kpoints(atoms: Atoms, density: float) -> list[int]:
    """
    Calculates the Monkhorst-Pack grid based on K-point density.

    Args:
        atoms: The atomic structure.
        density: K-points density factor.

    Returns:
        List[int]: The [k1, k2, k3] grid.
    """
    # Use ASE's cell.reciprocal() which returns vectors without 2pi
    # We multiply by 2pi to match standard physics convention assumed by density
    b_matrix = atoms.cell.reciprocal()
    kpts = []
    for i in range(3):
        b_len = np.linalg.norm(b_matrix[i]) * 2 * np.pi
        # Formula from Spec: k_i = max(1, round(|b_i| * density))
        k = max(1, round(b_len * density))
        kpts.append(k)
    return kpts


def is_magnetic(atoms: Atoms) -> bool:
    """
    Checks if the system contains magnetic elements.

    Args:
        atoms: The atomic structure.

    Returns:
        bool: True if magnetic elements (Fe, Ni, Co) are present.
    """
    magnetic_elements = {"Fe", "Ni", "Co"}
    symbols = set(atoms.get_chemical_symbols())
    return not symbols.isdisjoint(magnetic_elements)


def get_sssp_pseudopotentials(elements: list[str]) -> dict[str, str]:
    """
    Maps elements to their SSSP pseudopotential filenames.

    Args:
        elements: List of chemical symbols.

    Returns:
        Dict[str, str]: Map of element -> filename.
    """
    # Placeholder map
    mapping = {
        "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
        "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
        "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Fe": "Fe.pbe-spn-rrkjus_psl.1.0.0.UPF",
        "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    }

    return {el: mapping.get(el, f"{el}.pbe-standard.UPF") for el in elements}
