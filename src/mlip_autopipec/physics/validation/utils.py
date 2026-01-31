import ase.build
from pathlib import Path
import ase.data
from ase.calculators.lammpsrun import LAMMPS
from mlip_autopipec.domain_models.config import PotentialConfig, ValidationConfig
from mlip_autopipec.domain_models.structure import Structure


def get_calculator(potential_path: Path, config: PotentialConfig, lammps_command: str = "lammps") -> LAMMPS:
    """
    Creates an ASE LAMMPS calculator for the given potential.

    Args:
        potential_path: Path to the .yace potential file.
        config: Potential configuration.
        lammps_command: Command to run LAMMPS (e.g. 'lmp_serial').
                        Passed to ASE as a string. ASE uses subprocess/os.system depending on version,
                        but typically expects a string command line.
                        We rely on ASE's internal handling, but assume lammps_command comes from
                        validated LammpsConfig.
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
    # We use explicit command string from config
    calc = LAMMPS(
        command=lammps_command,
        pair_style=pair_style,
        pair_coeff=pair_coeffs,
        keep_tmp_files=False,
        specorder=sorted_elements,
    )

    return calc


def get_reference_structure(validation_config: ValidationConfig, potential_config: PotentialConfig) -> Structure:
    """
    Generate reference structure based on ValidationConfig.
    Raises ValueError if reference parameters are missing.
    """
    element = potential_config.elements[0]
    crystal = validation_config.ref_crystal_structure
    a = validation_config.ref_lattice_constant

    if crystal is None or a is None:
        raise ValueError(
            "ValidationConfig missing 'ref_crystal_structure' or 'ref_lattice_constant'. "
            "These are required for validation (EOS, Elasticity, Phonons)."
        )

    # Use ase.build.bulk
    try:
        atoms = ase.build.bulk(element, crystal, a=a) # type: ignore[no-untyped-call]
    except Exception as e:
        raise ValueError(f"Could not generate reference structure for {element} with {crystal}, a={a}: {e}")

    return Structure.from_ase(atoms)
