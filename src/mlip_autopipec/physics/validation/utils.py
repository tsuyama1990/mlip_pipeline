from pathlib import Path

from ase.calculators.lammpsrun import LAMMPS
from ase.data import atomic_masses, atomic_numbers


def get_lammps_calculator(potential_path: Path, elements: list[str] = ["Si"]) -> LAMMPS:
    """
    Returns an ASE LAMMPS calculator configured for the PACE potential.
    """
    # pair_style pace
    # pair_coeff * * potential.yace Si

    # We need to handle potential path.
    # LAMMPS calculator runs in a temp dir usually.
    # We need to ensure potential file is accessible.

    # Command: pair_style pace
    # But we might want hybrid/overlay zbl.
    # For now, simplistic implementation.

    potential_str = str(potential_path.absolute())

    # Spec mapping
    specorder = elements

    # Basic settings
    parameters = {
        "pair_style": "pace",
        "pair_coeff": [f"* * {potential_str} {' '.join(elements)}"],
        "mass": [
            f"{i + 1} {atomic_masses[atomic_numbers[el]]}"
            for i, el in enumerate(elements)
        ],
    }

    # We assume 'pace' pair style is available in the LAMMPS executable used by ASE.
    # ASE looks for 'lmp_serial' or 'lmp_mpi'.

    calc = LAMMPS(
        specorder=specorder,
        parameters=parameters,
        files=[potential_str],
        keep_tmp_files=False,
    )
    return calc
