import json
import logging

import numpy as np
from ase import Atoms

from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


def apply_strain(atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
    """
    Applies a generic strain tensor to the atoms object.
    """
    try:
        strained = atoms.copy() # type: ignore[no-untyped-call]
        cell = strained.get_cell()

        # Deformation gradient F = I + epsilon
        deformation = np.eye(3) + strain_tensor

        # New cell vectors
        new_cell = np.dot(cell, deformation)

        strained.set_cell(new_cell, scale_atoms=True)

        strained.info["config_type"] = "strain"
        # Store as string for ASE DB compatibility
        strained.info["strain_tensor"] = json.dumps(strain_tensor.tolist())

    except Exception as e:
        msg = f"Failed to apply strain: {e}"
        raise GeneratorError(msg) from e

    return strained


def apply_rattle(atoms: Atoms, sigma: float) -> Atoms:
    """
    Applies random thermal displacement (rattling) to atomic positions.
    """
    try:
        rattled = atoms.copy() # type: ignore[no-untyped-call]
        delta = np.random.normal(0, sigma, atoms.positions.shape)
        rattled.positions += delta

        rattled.info["config_type"] = "rattle"
        rattled.info["rattle_sigma"] = sigma

    except Exception as e:
        msg = f"Failed to apply rattle: {e}"
        raise GeneratorError(msg) from e

    return rattled
