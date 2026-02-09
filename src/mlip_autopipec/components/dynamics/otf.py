import logging

logger = logging.getLogger(__name__)

OTF_CHECK_TEMPLATE = """
def check_uncertainty(atoms, threshold):
    \"\"\"
    Check if the structure has high uncertainty.

    Args:
        atoms: ASE Atoms object with 'uncertainty' in info or arrays.
        threshold: The max allowed uncertainty (gamma).

    Returns:
        bool: True if uncertainty > threshold.
    \"\"\"
    gamma = atoms.info.get('uncertainty', 0.0)
    # Check max per-atom uncertainty if available in arrays
    if 'uncertainty' in atoms.arrays:
        gamma = max(gamma, atoms.arrays['uncertainty'].max())

    if gamma > threshold:
        return True
    return False
"""


def get_otf_check_code() -> str:
    return OTF_CHECK_TEMPLATE


def generate_lammps_otf_commands(threshold: float) -> list[str]:
    """
    Generate LAMMPS commands for OTF monitoring.
    Assumes 'pace' compute is available as 'compute pace all pace ...'
    """
    # compute ID group-ID pace filename elements...
    # We assume 'pace' compute calculates per-atom gamma (extrapolation grade).
    # Usually PACE compute outputs: energy, stress, forces... and maybe gamma?
    # Spec says: "The engine automatically configures LAMMPS... compute pace and fix halt".

    # Typical MTP/ACE compute syntax varies. Assuming user-pace provides 'compute pace'.
    # If not, we might need 'compute entropy/atom' or similar.
    # But let's stick to 'compute pace'.
    # Let's assume the compute ID is 'pace_compute'.
    # And we access the global scalar (max gamma) or per-atom max.

    # Example: compute 1 all pace ...
    # variable max_gamma equal c_1[index_of_gamma]

    # Without specific documentation on the `pace` compute output vector,
    # we assume it outputs [potential_energy, max_gamma, ...] or similar.

    # Placeholder implementation based on SPEC "v_max_gamma > ${thresh}"

    return [
        # define variable for threshold
        f"variable threshold equal {threshold}",
        f"fix halt_otf all halt 10 v_max_gamma > {threshold} error hard",
    ]
