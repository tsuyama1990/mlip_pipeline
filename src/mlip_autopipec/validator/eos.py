from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


class EOSResults(NamedTuple):
    """
    Results of Equation of State fit.

    Attributes:
        E0: Equilibrium energy (eV)
        V0: Equilibrium volume (A^3)
        B0: Bulk modulus (GPa)
        B0_prime: Pressure derivative of bulk modulus
    """
    E0: float
    V0: float
    B0: float
    B0_prime: float

class EOSAnalyzer:
    """Analyzes Equation of State."""

    def _birch_murnaghan(
        self,
        volume: float | NDArray[np.float64],
        e0: float,
        v0: float,
        b0: float,
        b0_prime: float
    ) -> float | NDArray[np.float64]:
        """
        Birch-Murnaghan equation of state.

        Args:
            volume: Volume (A^3)
            e0: Equilibrium energy (eV)
            v0: Equilibrium volume (A^3)
            b0: Bulk modulus (eV/A^3)
            b0_prime: Pressure derivative of bulk modulus (dimensionless)

        Returns:
            Energy (eV)
        """
        eta = (v0 / volume)**(2/3)
        return e0 + 9 * v0 * b0 / 16 * (
            (eta - 1)**3 * b0_prime + (eta - 1)**2 * (6 - 4 * eta)
        )

    def fit_birch_murnaghan(
        self,
        volumes: list[float] | NDArray[Any],
        energies: list[float] | NDArray[Any]
    ) -> EOSResults:
        """
        Fits Energy-Volume data to Birch-Murnaghan EOS.

        Args:
            volumes: List or array of volumes (A^3)
            energies: List or array of energies (eV)

        Returns:
            EOSResults containing fitted parameters.

        Raises:
            ValueError: If input data is insufficient or invalid.
        """
        v = np.array(volumes)
        e = np.array(energies)

        if len(v) < 4:
            msg = "Need at least 4 data points for EOS fit."
            raise ValueError(msg)

        if len(v) != len(e):
            msg = "Volumes and energies must have same length."
            raise ValueError(msg)

        # Initial guess
        # Sort by energy to find minimum roughly
        min_idx = np.argmin(e)
        v0_guess = v[min_idx]
        e0_guess = e[min_idx]
        b0_guess = 1.0  # approx 160 GPa, reasonable guess
        b0_prime_guess = 4.0  # typical value

        p0 = [e0_guess, v0_guess, b0_guess, b0_prime_guess]

        try:
            popt, pcov = curve_fit(self._birch_murnaghan, v, e, p0=p0)
        except RuntimeError as err:
            msg = f"EOS fit failed: {err}"
            raise ValueError(msg) from err

        e0_fit, v0_fit, b0_fit, b0_prime_fit = popt

        # Convert B0 from eV/A^3 to GPa
        b0_gpa = b0_fit * 160.21766208

        return EOSResults(
            E0=float(e0_fit),
            V0=float(v0_fit),
            B0=float(b0_gpa),
            B0_prime=float(b0_prime_fit)
        )
