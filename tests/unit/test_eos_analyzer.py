from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from mlip_autopipec.validator.eos import EOSAnalyzer, EOSResults


def test_eos_analyzer_fit() -> None:
    """Test Birch-Murnaghan fit with synthetic data."""
    # Synthetic data for Cu (approximate)
    # V0 = 11.0, B0 = 130 GPa, B0_prime = 5.0
    v0_true = 11.0
    b0_true = 130.0  # GPa
    b0_prime_true = 5.0
    e0_true = -3.5

    def bm_eqn(
        volume: float | NDArray[Any],
        e0: float,
        v0: float,
        b0: float,
        b0_prime: float
    ) -> float | NDArray[Any]:
        """Birch-Murnaghan equation of state."""
        eta = (v0 / volume)**(2/3)
        return e0 + 9 * v0 * b0 / 16 * (
            (eta - 1)**3 * b0_prime + (eta - 1)**2 * (6 - 4 * eta)
        )

    volumes: NDArray[Any] = np.linspace(0.9 * v0_true, 1.1 * v0_true, 10)

    # Convert B0 to eV/A^3 for the equation
    b0_ev_a3 = b0_true / 160.21766208
    energies_raw = bm_eqn(volumes, e0_true, v0_true, b0_ev_a3, b0_prime_true)
    energies = cast(NDArray[Any], energies_raw)

    analyzer = EOSAnalyzer()
    results = analyzer.fit_birch_murnaghan(volumes, energies)

    assert isinstance(results, EOSResults)
    assert pytest.approx(v0_true, rel=1e-3) == results.V0
    assert pytest.approx(b0_true, rel=1e-2) == results.B0 # B0 in GPa
    assert pytest.approx(e0_true, rel=1e-3) == results.E0

def test_eos_analyzer_fit_failure() -> None:
    """Test behavior when fit fails."""
    analyzer = EOSAnalyzer()

    # Test with insufficient data points
    with pytest.raises(ValueError, match="Need at least 4 data points"):
        analyzer.fit_birch_murnaghan([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    # Test with unequal length
    with pytest.raises(ValueError, match="Volumes and energies must have same length"):
        analyzer.fit_birch_murnaghan([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])

    # Test with fitting error (garbage data that causes optimization failure)
    # This is covered by test_eos_analyzer_runtime_error using mocks.

def test_eos_analyzer_runtime_error() -> None:
    """Test RuntimeError handling during fit."""
    analyzer = EOSAnalyzer()

    with patch("mlip_autopipec.validator.eos.curve_fit", side_effect=RuntimeError("Mock Failure")), \
         pytest.raises(ValueError, match="EOS fit failed: Mock Failure"):
        analyzer.fit_birch_murnaghan([1,2,3,4], [1,2,3,4])
