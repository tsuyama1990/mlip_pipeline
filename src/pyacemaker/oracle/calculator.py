"""Calculator factory for DFT simulations."""

from typing import Any

from ase.calculators.calculator import BaseCalculator
from ase.calculators.espresso import Espresso, EspressoProfile

from pyacemaker.core.config import CONSTANTS, DFTConfig


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def create_calculator(
    config: DFTConfig, attempt: int = 0, directory: str | None = None
) -> BaseCalculator:
    """Create an ASE calculator based on configuration and attempt number.

    Args:
        config: The DFT configuration.
        attempt: The retry attempt number (0-based).
        directory: Working directory for the calculator.

    Returns:
        Configured ASE Calculator.

    Raises:
        NotImplementedError: If the configured DFT code is not supported.
        ValueError: If input parameters are invalid or contain restricted keys.

    """
    if config.code != "quantum_espresso":
        msg = f"DFT code '{config.code}' is not supported. Only 'quantum_espresso' is available."
        raise NotImplementedError(msg)

    # Base parameters
    input_data: dict[str, Any] = {
        "control": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "pseudo_dir": str(config.pseudo_dir),
            "tprnfor": True,
            "tstress": True,
        },
        "system": {
            "ecutwfc": CONSTANTS.default_dft_ecutwfc,
            "ecutrho": CONSTANTS.default_dft_ecutrho,
            "occupations": CONSTANTS.default_dft_occupations,
            "smearing": CONSTANTS.default_dft_smearing_method,
            "degauss": config.smearing,
        },
        "electrons": {
            "mixing_beta": CONSTANTS.default_dft_mixing_beta,
            "conv_thr": CONSTANTS.default_dft_conv_thr,
        },
    }

    # Override with user parameters (Secure)
    if config.parameters:
        if not isinstance(config.parameters, dict):
            msg = "Parameters must be a dictionary"
            raise ValueError(msg)

        # Validate allowed sections
        for key in config.parameters:
            if key.lower() not in CONSTANTS.dft_allowed_input_sections:
                msg = (
                    f"Security Error: Input section '{key}' is not allowed. "
                    f"Allowed sections: {CONSTANTS.dft_allowed_input_sections}"
                )
                raise ValueError(msg)

        _deep_update(input_data, config.parameters)

    # Adjust parameters based on attempt (Self-Healing)
    # This comes AFTER user overrides to ensure retry logic works on top of user settings
    if attempt > 0:
        # Reduce mixing beta for stability
        # We need to ensure 'electrons' exists if user messed it up, but input_data defaults have it
        if "electrons" not in input_data:
            input_data["electrons"] = {}

        current_beta = input_data["electrons"].get("mixing_beta", CONSTANTS.default_dft_mixing_beta)
        # Ensure current_beta is a float/number
        if not isinstance(current_beta, (int, float)):
            # Fallback if user provided weird type
            current_beta = CONSTANTS.default_dft_mixing_beta

        new_beta = max(0.1, current_beta - (0.1 * attempt))
        input_data["electrons"]["mixing_beta"] = new_beta

    # ASE Espresso calculator
    # Use EspressoProfile for newer ASE versions
    profile = EspressoProfile(  # type: ignore[no-untyped-call]
        command=config.command, pseudo_dir=str(config.pseudo_dir)
    )

    # We use type: ignore because ASE types are often missing/incomplete
    calc = Espresso(  # type: ignore[no-untyped-call]
        profile=profile,
        pseudopotentials=config.pseudopotentials,
        tstress=True,
        tprnfor=True,
        kspacing=config.kspacing,
        input_data=input_data,
        directory=directory or ".",
    )

    if not isinstance(calc, BaseCalculator):
        msg = f"Created object is not an ASE Calculator: {type(calc)}"
        raise TypeError(msg)

    return calc
