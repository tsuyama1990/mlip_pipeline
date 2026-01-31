from pathlib import Path
import os
from typing import Any
import ase.data
from ase.calculators.lammpsrun import LAMMPS
from mlip_autopipec.domain_models.config import PotentialConfig

def get_lammps_calculator(
    potential_path: Path,
    potential_config: PotentialConfig,
    lammps_command: str = "lmp",
    working_dir: Path = Path("/tmp")
) -> LAMMPS:
    """
    Configures and returns an ASE LAMMPS calculator.
    """
    # Ensure working dir exists
    working_dir.mkdir(parents=True, exist_ok=True)

    # Check if lammps command is set in env, otherwise use provided
    cmd = os.environ.get("LAMMPS_COMMAND", lammps_command)

    # Determine pair style/coeff
    pair_style = ""
    pair_coeff = [] # List of lines

    # We need to know elements to write pair_coeff correctly.
    # ASE LAMMPS calculator handles writing data file, but we need to specify interaction.
    # The 'specorder' argument in LAMMPS calculator ensures species match types.
    elements = sorted(potential_config.elements)

    if potential_config.pair_style == "hybrid/overlay":
        zbl_in = potential_config.zbl_inner_cutoff
        zbl_out = potential_config.zbl_outer_cutoff

        pair_style = f"hybrid/overlay pace zbl {zbl_in} {zbl_out}"

        # PACE
        pot_file = str(potential_path.resolve())
        elem_str = " ".join(elements)
        pair_coeff.append(f"* * pace {pot_file} {elem_str}")

        # ZBL
        # We assume elements are ordered as in 'specorder'
        for i, el1 in enumerate(elements):
            z1 = ase.data.atomic_numbers[el1]
            for j, el2 in enumerate(elements):
                if j < i:
                    continue
                z2 = ase.data.atomic_numbers[el2]
                pair_coeff.append(f"{i+1} {j+1} zbl {z1} {z2}")

    else:
        pair_style = "pace"
        pot_file = str(potential_path.resolve())
        elem_str = " ".join(elements)
        pair_coeff.append(f"* * pace {pot_file} {elem_str}")

    parameters = {
        "pair_style": pair_style,
        "pair_coeff": pair_coeff,
        "mass": [ase.data.atomic_masses[ase.data.atomic_numbers[el]] for el in elements],
        "specorder": elements,
        "command": cmd,
    }

    # We use 'tmp_dir' or 'label' to control where it runs?
    # ASE lammpsrun uses 'label' as prefix.
    # We prefer to run in a specific directory.
    # We can pass 'tmp_dir' if using newer ASE, or set CWD before creation?
    # ASE's LAMMPS calculator is a FileIOCalculator.

    calc = LAMMPS(
        label="val",
        tmp_dir=str(working_dir),
        keep_tmp_files=True, # Debugging
        files=[str(potential_path.resolve())], # Copy potential file? ACE file needs to be there or absolute path.
        **parameters
    )

    return calc
