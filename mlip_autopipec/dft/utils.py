from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.exceptions import DFTException


def get_kpoints(atoms: Atoms, density: float) -> list[int]:
    """
    Calculates Monkhorst-Pack grid based on density.
    ki = max(1, round(|bi| * density))
    """
    # atoms.get_reciprocal_cell() returns 2pi * (inv(cell).T)
    # The magnitude of reciprocal vectors.
    recip_cell = atoms.cell.reciprocal()
    b_norms = np.linalg.norm(recip_cell, axis=1)

    kpoints = []
    for b in b_norms:
        k = max(1, round(b * density))
        kpoints.append(k)

    return kpoints


def is_magnetic(atoms: Atoms) -> bool:
    """
    Checks if magnetic elements (Fe, Ni, Co) are present.
    """
    magnetic_elements = {"Fe", "Ni", "Co"}
    symbols = set(atoms.get_chemical_symbols())
    return not symbols.isdisjoint(magnetic_elements)


def get_sssp_pseudopotentials(atoms: Atoms, pseudo_dir: Path) -> dict[str, str]:
    """
    Maps elements to UPF files.
    Looks for {Symbol}.upf or {Symbol}.*.upf in pseudo_dir.
    """
    pseudos = {}
    unique_symbols = sorted(set(atoms.get_chemical_symbols()))

    for sym in unique_symbols:
        # Search patterns: exact match or with suffix
        candidates = list(pseudo_dir.glob(f"{sym}.upf"))
        if not candidates:
            candidates = list(pseudo_dir.glob(f"{sym}.*.upf"))

        if not candidates:
            msg = f"No pseudopotential found for {sym} in {pseudo_dir}"
            raise DFTException(msg)

        # Pick the first one (usually sorted alphabetically)
        candidates.sort()
        pseudos[sym] = candidates[0].name

    return pseudos
