from pathlib import Path

import ase.data
from ase.calculators.lammpsrun import LAMMPS
from mlip_autopipec.domain_models.config import PotentialConfig

def get_lammps_calculator(
    potential_path: Path,
    config: PotentialConfig,
    work_dir: Path,
    command: str = "lmp"
) -> LAMMPS:
    """
    Factory to create an ASE LAMMPS calculator configured with the potential.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    unique_elements = config.elements

    pair_style = ""
    pair_coeff = []

    # Check potential path
    abs_pot_path = potential_path.resolve()

    if config.pair_style == "hybrid/overlay":
         zbl_in = config.zbl_inner_cutoff
         zbl_out = config.zbl_outer_cutoff
         pair_style = f"hybrid/overlay pace zbl {zbl_in} {zbl_out}"

         # Pace
         # We use the filename because ASE copies it to tmp_dir
         pot_file_name = abs_pot_path.name
         elem_str = " ".join(unique_elements)
         pair_coeff.append(f"* * pace {pot_file_name} {elem_str}")

         # ZBL
         for i, el1 in enumerate(unique_elements):
             z1 = ase.data.atomic_numbers[el1]
             for j, el2 in enumerate(unique_elements):
                 if j < i:
                     continue
                 z2 = ase.data.atomic_numbers[el2]
                 pair_coeff.append(f"{i+1} {j+1} zbl {z1} {z2}")

    else:
         pair_style = "pace"
         pot_file_name = abs_pot_path.name
         elem_str = " ".join(unique_elements)
         pair_coeff.append(f"* * pace {pot_file_name} {elem_str}")

    calc = LAMMPS(
        label="val",
        tmp_dir=str(work_dir),
        pair_style=pair_style,
        pair_coeff=pair_coeff,
        specorder=unique_elements,
        files=[str(abs_pot_path)],
        keep_tmp_files=True,
        command=command
    )

    return calc
