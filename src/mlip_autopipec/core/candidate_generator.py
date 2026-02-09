import logging
import warnings

import numpy as np

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


def generate_local_candidates(
    structure: Structure, n_candidates: int = 20, rattle_strength: float = 0.05
) -> list[Structure]:
    """
    DEPRECATED: Use Generator.enhance() instead.

    Generate local candidates around a structure using random displacement (Rattle).

    Args:
        structure: The seed structure (e.g. halted structure).
        n_candidates: Number of candidates to generate.
        rattle_strength: Magnitude of random displacement in Angstroms.

    Returns:
        List of generated Structure objects, including the original (anchor).
    """
    warnings.warn(
        "generate_local_candidates is deprecated. Use Generator.enhance() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    candidates = [structure]  # Always include the anchor

    # We use the structure's own logic if available, or just manipulate positions
    # Structure.positions is a numpy array.

    for _i in range(n_candidates):
        # Deep copy to ensure independence
        new_struct = structure.model_deep_copy()

        # Apply random displacement
        displacement = np.random.uniform(
            -rattle_strength, rattle_strength, size=new_struct.positions.shape
        )
        new_struct.positions += displacement

        # Update tags
        new_struct.tags["provenance"] = "local_candidate"
        new_struct.tags["parent_halt"] = structure.tags.get("provenance", "unknown")

        # Reset labels/uncertainty as they are now unknown
        new_struct.energy = None
        new_struct.forces = None
        new_struct.stress = None
        new_struct.uncertainty = None

        candidates.append(new_struct)

    return candidates
