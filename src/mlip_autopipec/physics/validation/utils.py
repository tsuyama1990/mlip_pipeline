import os
from pathlib import Path

import ase.data
from ase.calculators.lammpsrun import LAMMPS

from mlip_autopipec.domain_models.config import PotentialConfig


def get_validation_calculator(
    potential_path: Path, potential_config: PotentialConfig, work_dir: Path
) -> LAMMPS:
    """
    Configures an ASE LAMMPS calculator for validation.
    """
    elements = potential_config.elements
    specorder = elements  # Enforces mapping: elements[0] -> type 1

    # Define pair style and coefficients
    pair_style = ""
    pair_coeff = []

    if potential_config.pair_style == "hybrid/overlay":
        zbl_in = potential_config.zbl_inner_cutoff
        zbl_out = potential_config.zbl_outer_cutoff
        pair_style = f"hybrid/overlay pace zbl {zbl_in} {zbl_out}"

        # Pace
        # pair_coeff * * pace potential.yace Element1 Element2 ...
        elem_str = " ".join(elements)
        # Using .resolve() for absolute path is safer for LAMMPS
        pot_abs = potential_path.resolve()
        pair_coeff.append(f"* * pace {pot_abs} {elem_str}")

        # ZBL
        # pair_coeff i j zbl Zi Zj
        for i, el1 in enumerate(elements):
            z1 = ase.data.atomic_numbers[el1]
            for j, el2 in enumerate(elements):
                if j < i:
                    continue
                z2 = ase.data.atomic_numbers[el2]
                pair_coeff.append(f"{i + 1} {j + 1} zbl {z1} {z2}")

    else:
        # Just pace
        pair_style = "pace"
        elem_str = " ".join(elements)
        pot_abs = potential_path.resolve()
        pair_coeff.append(f"* * pace {pot_abs} {elem_str}")

    cmd = os.environ.get("LAMMPS_COMMAND", "lmp")

    return LAMMPS(
        command=cmd,
        label="val",
        directory=str(work_dir),
        specorder=specorder,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
        # We don't list files here because we use absolute path in pair_coeff.
        # But providing it ensures ASE doesn't complain about missing files if it checks.
        files=[str(potential_path.resolve())],
        keep_tmp_files=False,
    )
