import logging

from ase.data import atomic_numbers

from mlip_autopipec.domain_models.config import PhysicsBaselineConfig
from mlip_autopipec.domain_models.potential import Potential

logger = logging.getLogger(__name__)


def generate_pair_style(
    potential: Potential, baseline: PhysicsBaselineConfig | None
) -> tuple[str, str]:
    """
    Generate LAMMPS pair_style and pair_coeff commands for hybrid potentials.

    Args:
        potential: The ML potential (ACE/PACE).
        baseline: The physics baseline configuration (ZBL/LJ).

    Returns:
        tuple[str, str]: (pair_style command, pair_coeff commands)
    """
    # Base PACE style
    # Assuming user-pace is installed as 'pace'
    pace_style = "pace"
    pace_coeff_line = f"pair_coeff * * {pace_style} {potential.path} {' '.join(potential.species)}"

    if not baseline:
        return f"pair_style {pace_style}", pace_coeff_line

    if baseline.type == "zbl":
        # Hybrid overlay
        pair_style = (
            f"pair_style hybrid/overlay {pace_style} zbl 0.5 2.0"  # Default cutoffs for ZBL mixing?
        )
        # ZBL usually is very short range. 0.5 inner, 2.0 outer is reasonable for overlay?
        # Actually ZBL is purely repulsive core.
        # LAMMPS syntax: pair_style zbl cut_inner cut_outer
        # But for hybrid/overlay, we specify sub-styles.
        # Let's use params if available
        inner = baseline.params.get("inner_cutoff", 0.5)
        outer = baseline.params.get("outer_cutoff", 1.2)

        pair_style = f"pair_style hybrid/overlay {pace_style} zbl {inner} {outer}"

        coeffs = [pace_coeff_line]

        # Generate ZBL coeffs for all pairs
        species = potential.species
        n_types = len(species)

        for i in range(n_types):
            for j in range(i, n_types):
                el1 = species[i]
                el2 = species[j]
                z1 = atomic_numbers[el1]
                z2 = atomic_numbers[el2]

                # LAMMPS type indices are 1-based
                idx1 = i + 1
                idx2 = j + 1

                # pair_coeff I J zbl Z1 Z2
                coeffs.append(f"pair_coeff {idx1} {idx2} zbl {z1} {z2}")

        return pair_style, "\n".join(coeffs)

    if baseline.type == "lj":
        # TODO: Implement LJ logic if needed
        logger.warning("LJ baseline not fully implemented, using PACE only.")
        return f"pair_style {pace_style}", pace_coeff_line

    return f"pair_style {pace_style}", pace_coeff_line
