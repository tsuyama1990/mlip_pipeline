from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.validation import ElasticityConfig
from mlip_autopipec.validation.elasticity import ElasticityValidator


def test_elasticity_check_stability_cubic():
    # Stable Cubic
    C = np.zeros((6, 6))
    C11, C12, C44 = 100.0, 50.0, 30.0
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[0, 1] = C[0, 2] = C[1, 2] = C[1, 0] = C[2, 0] = C[2, 1] = C12
    C[3, 3] = C[4, 4] = C[5, 5] = C44

    assert ElasticityValidator.check_born_stability(C, system_type="cubic") is True


def test_elasticity_check_stability_cubic_unstable_shear():
    # Unstable Shear (C11 - C12 < 0)
    C = np.zeros((6, 6))
    C11, C12, C44 = 100.0, 120.0, 30.0
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[0, 1] = C[0, 2] = C[1, 2] = C[1, 0] = C[2, 0] = C[2, 1] = C12
    C[3, 3] = C[4, 4] = C[5, 5] = C44

    assert ElasticityValidator.check_born_stability(C, system_type="cubic") is False


def test_elasticity_check_stability_cubic_unstable_bulk():
    # Unstable Bulk (C11 + 2C12 < 0)
    C = np.zeros((6, 6))
    C11, C12, C44 = 100.0, -60.0, 30.0
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[0, 1] = C[0, 2] = C[1, 2] = C[1, 0] = C[2, 0] = C[2, 1] = C12
    C[3, 3] = C[4, 4] = C[5, 5] = C44

    assert ElasticityValidator.check_born_stability(C, system_type="cubic") is False


def test_elasticity_calculation_mock():
    config = ElasticityConfig(strain_max=0.01, num_points=2)
    validator = ElasticityValidator(config)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    calculator = MagicMock()

    # Mock get_stress returns different values for different strains?
    # It's hard to mock realistic stress responses for 6 deformations * points.
    # Instead, we mock internal calculate_elastic_tensor method if we were doing integration test,
    # but here we want to test that 'validate' calls 'calculate_elastic_tensor' and uses the result.

    # We will use monkeypatch to mock the internal calculation method for this test
    # assuming we split the logic.

    C_stable = np.zeros((6, 6))
    C11, C12, C44 = 100.0, 50.0, 30.0
    C_stable[0, 0] = C_stable[1, 1] = C_stable[2, 2] = C11
    C_stable[0, 1] = C_stable[0, 2] = C_stable[1, 2] = C12
    C_stable[3, 3] = C_stable[4, 4] = C_stable[5, 5] = C44

    # Determine system type logic (e.g. assume cubic for simplicity or check lattice)
    # We'll assume the implementation detects cubic.

    with pytest.MonkeyPatch.context() as m:
        m.setattr(validator, "calculate_elastic_tensor", lambda a, c: C_stable)
        result = validator.validate(atoms, calculator)
        assert result is True

    # Test Unstable
    C_unstable = np.zeros((6, 6))
    C11, C12, C44 = 100.0, 120.0, 30.0  # Unstable
    C_unstable[0, 0] = C11
    C_unstable[0, 1] = C12

    with pytest.MonkeyPatch.context() as m:
        m.setattr(validator, "calculate_elastic_tensor", lambda a, c: C_unstable)
        result = validator.validate(atoms, calculator)
        assert result is False
