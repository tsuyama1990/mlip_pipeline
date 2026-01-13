"""Utilities for preparing and parsing Quantum Espresso calculations."""

import json
import pkgutil

import numpy as np
from ase import Atoms


def get_sssp_recommendations(atoms: Atoms) -> dict[str, dict[str, float | str]]:
    """
    Looks up SSSP library recommendations for the elements in the Atoms object.

    Args:
        atoms: An ASE Atoms object.

    Returns:
        A dictionary containing the recommended pseudopotential and cutoffs for each element.
    """
    data_bytes = pkgutil.get_data("mlip_autopipec.utils", "sssp_data.json")
    if not data_bytes:
        error_msg = "Could not find sssp_data.json"
        raise FileNotFoundError(error_msg)
    sssp_data = json.loads(data_bytes.decode("utf-8"))

    recommendations = {}
    elements = set(atoms.get_chemical_symbols())  # type: ignore
    for element in elements:
        if element not in sssp_data:
            error_msg = f"No SSSP recommendation found for element: {element}"
            raise ValueError(error_msg)
        recommendations[element] = sssp_data[element]
    return recommendations


def get_kpoints(atoms: Atoms, k_point_density: float = 64.0) -> tuple[int, int, int]:
    """
    Calculates the k-point mesh for a given structure based on a target density.

    Args:
        atoms: An ASE Atoms object.
        k_point_density: The target number of k-points per reciprocal atom (kpra).

    Returns:
        A tuple of 3 integers representing the k-point mesh (nkx, nky, nkz).
    """
    reciprocal_lattice = atoms.get_reciprocal_cell()
    k_points = np.ceil(
        np.linalg.norm(reciprocal_lattice, axis=1) * k_point_density ** (1 / 3)
    ).astype(int)
    k_points[k_points == 0] = 1
    return tuple(k_points)


def is_magnetic(atoms: Atoms) -> bool:
    """Heuristic to determine if a system is likely to be magnetic."""
    magnetic_elements = {"Fe", "Ni", "Co", "Mn", "Cr"}
    elements = set(atoms.get_chemical_symbols())  # type: ignore
    return not elements.isdisjoint(magnetic_elements)


def is_metal(atoms: Atoms) -> bool:
    """Heuristic to determine if a system is likely to be metallic."""
    common_metals = {"Fe", "Ni", "Co", "Cu", "Ag", "Au", "Pt", "Al", "Ti"}
    common_insulators = {"Si", "C", "O", "N", "S", "Cl"}
    elements = set(atoms.get_chemical_symbols())  # type: ignore

    if not elements.isdisjoint(common_metals):
        return True
    return len(elements) == 1 and elements.pop() not in common_insulators
