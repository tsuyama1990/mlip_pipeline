import json
import logging

import numpy as np
from ase import Atoms

from mlip_autopipec.exceptions import GeneratorError

logger = logging.getLogger(__name__)


def apply_strain(atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
    """
    Applies a generic strain tensor to the atoms object.

    Args:
        atoms (Atoms): The structure to strain.
        strain_tensor (np.ndarray): A 3x3 symmetric strain tensor (epsilon).

    Returns:
        Atoms: The new strained structure with updated cell dimensions.

    Raises:
        GeneratorError: If matrix operations fail or inputs are invalid.
    """
    if not isinstance(strain_tensor, np.ndarray) or strain_tensor.shape != (3, 3):
        msg = "Strain tensor must be a 3x3 numpy array."
        raise GeneratorError(msg)

    try:
        strained = atoms.copy()
        cell = strained.get_cell()

        # Deformation gradient F = I + epsilon
        deformation = np.eye(3) + strain_tensor

        # New cell vectors
        new_cell = np.dot(cell, deformation)

        strained.set_cell(new_cell, scale_atoms=True)

        strained.info["config_type"] = "strain"
        # Store as string for ASE DB compatibility
        strained.info["strain_tensor"] = json.dumps(strain_tensor.tolist())

    except np.linalg.LinAlgError as e:
        msg = f"Matrix operation failed during strain application: {e}"
        logger.error(msg, exc_info=True)
        raise GeneratorError(msg) from e
    except GeneratorError:
        raise
    except Exception as e:
        msg = f"Failed to apply strain: {e}"
        logger.error(msg, exc_info=True)
        raise GeneratorError(msg) from e

    return strained


def apply_rattle(atoms: Atoms, sigma: float, rng: np.random.Generator | None = None) -> Atoms:
    """
    Applies random thermal displacement (rattling) to atomic positions.

    Args:
        atoms (Atoms): The structure to rattle.
        sigma (float): Standard deviation of the displacement in Angstroms.
        rng (np.random.Generator, optional): Random number generator for determinism.

    Returns:
        Atoms: The rattled structure.

    Raises:
        GeneratorError: If the operation fails.
    """
    if sigma < 0:
        msg = "Rattle standard deviation (sigma) must be non-negative."
        raise GeneratorError(msg)

    if rng is None:
        rng = np.random.default_rng()

    try:
        rattled = atoms.copy()

        delta = rng.normal(0, sigma, atoms.positions.shape)

        # Must set positions explicitly as .positions usually returns a copy
        rattled.set_positions(rattled.get_positions() + delta)

        rattled.info["config_type"] = "rattle"
        rattled.info["rattle_sigma"] = sigma

    except GeneratorError:
        raise
    except Exception as e:
        msg = f"Failed to apply rattle: {e}"
        logger.error(msg, exc_info=True)
        raise GeneratorError(msg) from e

    return rattled
