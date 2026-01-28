import logging
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def load_calculator(potential_path: Path) -> Calculator | Any:
    """
    Loads an ASE calculator for the given potential file.
    Supports .yace (ACE via pypacemaker) and .model (MACE).

    Args:
        potential_path: Path to the potential file.

    Returns:
        ASE Calculator instance.

    Raises:
        ValueError: If file extension is not supported.
        ImportError: If required library is missing.
        RuntimeError: If loading fails.
    """
    if not potential_path.exists():
        msg = f"Potential file not found: {potential_path}"
        raise FileNotFoundError(msg)

    suffix = potential_path.suffix.lower()

    if suffix in [".yace", ".ace"]:
        return _load_ace_calculator(potential_path)
    if suffix == ".model":
        return _load_mace_calculator(potential_path)

    msg = f"Unsupported potential format: {suffix}"
    raise ValueError(msg)


def _load_ace_calculator(potential_path: Path) -> Any:
    try:
        from pypacemaker import Calculator as PaceCalculator  # type: ignore

        return PaceCalculator(str(potential_path))
    except ImportError as e:
        logger.warning("pypacemaker not installed. Checking for alternative...")
        # Fallback or strict error?
        # Given we are in a specific environment, we should probably fail if primary tool is missing.
        msg = (
            "pypacemaker is required to validate .yace files. "
            "Please install it or use a different potential format."
        )
        raise ImportError(msg) from e


def _load_mace_calculator(potential_path: Path) -> Any:
    try:
        # Determine device
        import torch
        from mace.calculators import MACECalculator  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"

        return MACECalculator(
            model_paths=str(potential_path),
            device=device,
            default_dtype="float32",
        )
    except ImportError as e:
        msg = "mace-torch is required to validate .model files."
        raise ImportError(msg) from e
