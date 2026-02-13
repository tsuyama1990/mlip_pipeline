import numpy as np
import pytest

from mlip_autopipec.validator.eos import EOSAnalyzer, fit_birch_murnaghan


def test_fit_birch_murnaghan():
    # True parameters
    v0_true = 10.0
    e0_true = -5.0
    b0_gpa_true = 100.0
    b0_prime_true = 4.0

    EV_A3_TO_GPA = 160.21766208
    b0_ev_a3 = b0_gpa_true / EV_A3_TO_GPA

    def bm_eqn(v):
        vv0 = (v0_true / v) ** (2 / 3)
        return e0_true + (9 * v0_true * b0_ev_a3 / 16) * (
            (vv0 - 1) ** 3 * b0_prime_true + (vv0 - 1) ** 2 * (6 - 4 * vv0)
        )

    # Generate data
    volumes = np.linspace(0.8 * v0_true, 1.2 * v0_true, 20)
    energies = [bm_eqn(v) for v in volumes]

    # Fit
    results = fit_birch_murnaghan(list(volumes), energies)

    # Check results with tolerance
    assert results.volume == pytest.approx(v0_true, rel=1e-3)
    assert results.energy == pytest.approx(e0_true, rel=1e-3)
    assert results.bulk_modulus == pytest.approx(b0_gpa_true, rel=1e-3)
    assert results.bulk_modulus_derivative == pytest.approx(b0_prime_true, rel=1e-3)

def test_eos_analyzer_not_enough_points():
    analyzer = EOSAnalyzer()
    with pytest.raises(ValueError, match="At least 4 data points"):
        analyzer.analyze([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

def test_eos_analyzer_integration():
    analyzer = EOSAnalyzer()

    # Generate data
    v0_true = 15.0
    e0_true = -10.0
    b0_gpa_true = 70.0  # Al-like
    b0_prime_true = 4.0

    EV_A3_TO_GPA = 160.21766208
    b0_ev_a3 = b0_gpa_true / EV_A3_TO_GPA

    def bm_eqn(v):
        vv0 = (v0_true / v) ** (2 / 3)
        return e0_true + (9 * v0_true * b0_ev_a3 / 16) * (
            (vv0 - 1) ** 3 * b0_prime_true + (vv0 - 1) ** 2 * (6 - 4 * vv0)
        )

    volumes = np.linspace(13.0, 17.0, 10).tolist()
    energies = [bm_eqn(v) for v in volumes]

    metrics = analyzer.analyze(volumes, energies)

    assert metrics["equilibrium_volume"] == pytest.approx(v0_true, rel=1e-3)
    assert metrics["equilibrium_energy"] == pytest.approx(e0_true, rel=1e-3)
    assert metrics["bulk_modulus"] == pytest.approx(b0_gpa_true, rel=1e-3)
