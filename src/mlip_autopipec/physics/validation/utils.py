from pathlib import Path
import ase.data
from ase.calculators.lammpsrun import LAMMPS
from mlip_autopipec.domain_models.config import PotentialConfig


def get_calculator(potential_path: Path, config: PotentialConfig) -> LAMMPS:
    """
    Creates an ASE LAMMPS calculator for the given potential.
    """
    # Elements
    elements = config.elements
    # ASE sorts elements alphabetically for types usually when writing data file.
    # We must ensure we provide elements in the same order ASE expects (sorted).
    sorted_elements = sorted(elements)
    elem_str = " ".join(sorted_elements)

    pot_abs = potential_path.resolve()

    # We need to construct the pair_style and pair_coeff commands.
    # ASE LAMMPS calculator allows passing 'pair_style' and 'pair_coeff'.
    # For multiple pair_coeffs, we can pass a list.

    pair_style = ""
    pair_coeffs = []

    if config.pair_style == "hybrid/overlay":
        zbl_in = config.zbl_inner_cutoff
        zbl_out = config.zbl_outer_cutoff
        pair_style = f"hybrid/overlay pace zbl {zbl_in} {zbl_out}"

        # PACE coeff
        pair_coeffs.append(f"* * pace {pot_abs} {elem_str}")

        # ZBL coeffs
        # types are 1-indexed
        for i, el1 in enumerate(sorted_elements):
            z1 = ase.data.atomic_numbers[el1]
            for j, el2 in enumerate(sorted_elements):
                if j < i:
                    continue
                z2 = ase.data.atomic_numbers[el2]
                pair_coeffs.append(f"{i+1} {j+1} zbl {z1} {z2}")

    else:
        pair_style = "pace"
        pair_coeffs.append(f"* * pace {pot_abs} {elem_str}")

    # Create Calculator
    # We assume 'lammps' executable is in PATH.
    # We explicitly turn off 'echo' to reduce noise?
    calc = LAMMPS(
        command="lammps",
        pair_style=pair_style,
        pair_coeff=pair_coeffs,
        keep_tmp_files=False,
        specorder=sorted_elements,
    )

    return calc
