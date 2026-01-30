import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.infrastructure import io


def atoms_to_dataframe(atoms_list: List[Structure]) -> pd.DataFrame:
    """
    Convert a list of Structure objects to a Pandas DataFrame
    suitable for Pacemaker/TensorPotential.

    The DataFrame typically contains an 'ase_atoms' column.
    We also extract energy/forces/stress for clarity and checks.
    """
    data = []
    for structure in atoms_list:
        # Convert to ASE atoms
        # This reconstructs the Atoms object with info and arrays
        atoms = structure.to_ase()

        # Check if energy/forces are available in info/arrays
        energy = atoms.info.get("energy", None)
        forces = atoms.arrays.get("forces", None)
        stress = atoms.info.get("stress", None)

        # If not in info/arrays, try calculator (unlikely for deserialized Structure, but good for safety)
        if hasattr(atoms, "calc") and atoms.calc is not None:
            try:
                if energy is None:
                    energy = atoms.get_potential_energy()
                if forces is None:
                    forces = atoms.get_forces()
                if stress is None:
                    stress = atoms.get_stress()
            except Exception:
                pass

        # Ensure energy/forces are in the atoms object for Pacemaker
        if energy is not None:
            atoms.info["energy"] = energy
        if forces is not None:
            atoms.set_array("forces", np.array(forces)) # type: ignore[no-untyped-call]
        if stress is not None:
            atoms.info["stress"] = np.array(stress)

        data.append({
            "ase_atoms": atoms,
            "energy": energy,
        })

    return pd.DataFrame(data)


def save_dataset(atoms_list: List[Structure], path: Path) -> None:
    """
    Save list of structures to a .pckl.gzip file for Pacemaker.
    """
    df = atoms_to_dataframe(atoms_list)
    # Use protocol 4 or similar for compatibility if needed, but default is usually fine.
    df.to_pickle(str(path), compression="gzip")
