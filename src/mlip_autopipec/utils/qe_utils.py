import json
from pathlib import Path

import numpy as np
from ase import Atoms

# For now, we will use a simplified mock SSSP data.
# In a real implementation, this would be a more comprehensive database.
SSSP_DATA_PATH = Path(__file__).parent / "sssp_data.json"

def _load_sssp_data() -> dict:
    # In a real scenario, we'd have a robust way to package this data.
    # For now, create a dummy file if it doesn't exist.
    if not SSSP_DATA_PATH.exists():
        dummy_data = {
            "Si": {"cutoff_wfc": 30, "pseudopotential": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"},
            "Ni": {"cutoff_wfc": 40, "pseudopotential": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF"},
            "Fe": {"cutoff_wfc": 45, "pseudopotential": "Fe.pbe-n-rrkjus_psl.1.0.0.UPF"},
        }
        with SSSP_DATA_PATH.open("w") as f:
            json.dump(dummy_data, f, indent=2)
    with SSSP_DATA_PATH.open() as f:
        return json.load(f)

SSSP_DATA = _load_sssp_data()

def get_sssp_recommendations(atoms: Atoms) -> dict:
    """
    Gets SSSP recommendations for pseudopotentials and cutoffs.
    """
    elements = set(atoms.get_chemical_symbols())
    recommendations = {"pseudopotentials": {}, "cutoff_wfc": 0}
    max_cutoff = 0

    for element in elements:
        if element not in SSSP_DATA:
            raise ValueError(f"No SSSP data found for element: {element}")

        recommendations["pseudopotentials"][element] = SSSP_DATA[element]["pseudopotential"]
        max_cutoff = max(max_cutoff, SSSP_DATA[element]["cutoff_wfc"])

    recommendations["cutoff_wfc"] = max_cutoff
    return recommendations

def get_kpoints(atoms: Atoms, kpt_density: float = 6.0) -> tuple[int, int, int]:
    """
    Calculates k-point grid based on linear density heuristic.
    """
    cell = atoms.get_cell()
    reciprocal_cell = cell.reciprocal()
    kpts = [int(np.ceil(kpt_density * np.linalg.norm(vec))) for vec in reciprocal_cell]
    return tuple(max(1, k) for k in kpts)
