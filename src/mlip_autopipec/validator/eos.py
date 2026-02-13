import logging
from typing import NamedTuple, Any

import numpy as np

logger = logging.getLogger(__name__)

class EOSResults(NamedTuple):
    """Results from EOS fitting."""
    volume: float  # Equilibrium volume (Angstrom^3)
    energy: float  # Equilibrium energy (eV)
    bulk_modulus: float  # Bulk modulus (GPa)
    bulk_modulus_derivative: float  # Pressure derivative of bulk modulus

def fit_birch_murnaghan(volumes: list[float], energies: list[float]) -> EOSResults:
    """
    Fits Energy-Volume data to the Birch-Murnaghan equation of state.

    E(V) = E0 + 9*V0*B0/16 * {[(V0/V)^(2/3) - 1]^3 * B0' + [(V0/V)^(2/3) - 1]^2 * [6 - 4*(V0/V)^(2/3)]}

    Args:
        volumes: List of volumes in Angstrom^3.
        energies: List of energies in eV.

    Returns:
        EOSResults containing V0, E0, B0 (GPa), B0'.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        logger.exception("scipy is required for EOS fitting.")
        raise

    v = np.array(volumes)
    e = np.array(energies)

    # Initial guess
    # V0: mean volume
    # E0: min energy
    # B0: guess 100 GPa (converted to eV/A^3 later)
    # B0': guess 4.0

    # 160.21766208 is conversion factor from eV/A^3 to GPa
    EV_A3_TO_GPA = 160.21766208

    def birch_murnaghan(v: Any, e0: float, v0: float, b0: float, b0_prime: float) -> Any:
        """
        Birch-Murnaghan equation of state.
        b0 is in eV/Angstrom^3.
        """
        vv0 = (v0 / v) ** (2 / 3)
        return e0 + (9 * v0 * b0 / 16) * (
            (vv0 - 1) ** 3 * b0_prime + (vv0 - 1) ** 2 * (6 - 4 * vv0)
        )

    # Simple initial guess
    min_idx = np.argmin(e)
    v0_guess = v[min_idx]
    e0_guess = e[min_idx]
    b0_guess = 1.0  # ~160 GPa, reasonable starting point in eV/A^3
    b0_prime_guess = 4.0

    p0 = [e0_guess, v0_guess, b0_guess, b0_prime_guess]

    try:
        popt, pcov = curve_fit(birch_murnaghan, v, e, p0=p0)
    except RuntimeError as exc:
        msg = "EOS fitting failed to converge"
        logger.exception("EOS fit failed")
        # Return fallback or raise
        raise ValueError(msg) from exc

    e0, v0, b0_ev_a3, b0_prime = popt

    b0_gpa = b0_ev_a3 * EV_A3_TO_GPA

    return EOSResults(
        volume=v0,
        energy=e0,
        bulk_modulus=b0_gpa,
        bulk_modulus_derivative=b0_prime
    )


class EOSAnalyzer:
    """Analyzer for Equation of State."""

    def analyze(self, volumes: list[float], energies: list[float]) -> dict[str, float]:
        """
        Performs EOS analysis.

        Args:
            volumes: List of volumes.
            energies: List of energies.

        Returns:
            Dictionary with EOS parameters.
        """
        if len(volumes) < 4:
            msg = "At least 4 data points are required for EOS fitting."
            raise ValueError(msg)

        results = fit_birch_murnaghan(volumes, energies)

        return {
            "equilibrium_volume": results.volume,
            "equilibrium_energy": results.energy,
            "bulk_modulus": results.bulk_modulus,
            "bulk_modulus_derivative": results.bulk_modulus_derivative
        }
