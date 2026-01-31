from unittest.mock import MagicMock
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
import pytest

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator

class MockCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        # Return dummy stress
        # Stress is (xx, yy, zz, yz, xz, xy)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        # Minimal stress response to strain
        self.results["stress"] = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])

@pytest.fixture
def mock_atoms():
    return Atoms("Al", positions=[[0, 0, 0]], cell=[[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]], pbc=True)

@pytest.fixture
def validation_config():
    return ValidationConfig()

@pytest.fixture
def potential_config():
    return PotentialConfig(elements=["Al"], cutoff=5.0)

def test_elasticity_validator_pass(mock_atoms, validation_config, potential_config, tmp_path):
    validator = ElasticityValidator(
        potential_path=tmp_path / "pot.yace",
        config=validation_config,
        potential_config=potential_config
    )

    validator._get_calculator = MagicMock(return_value=MockCalculator())

    # Mock internal elasticity fitting if complex logic exists,
    # but ideally we want to test the logic.
    # For now, let's assume the calculate method runs without error and returns a result.
    # Since we mocked the calculator to return constant stress, the fit might be singular or weird,
    # but the validator should handle it or fail gracefully.
    # To make it pass, we might need a smarter mock or mock the `fit_elastic_constants` method if we separate it.

    # Let's mock the internal method that does the heavy lifting if we can't easily simulate physics
    # It returns (C_matrix, constants_dict)
    validator._calculate_elastic_tensor = MagicMock(return_value=(
        np.eye(6) * 100.0,
        {"C11": 100.0, "C12": 50.0, "C44": 30.0, "B": 70.0}
    ))

    result = validator.validate(reference_structure=mock_atoms)

    assert isinstance(result, ValidationResult)
    assert result.overall_status == "PASS"
    # Check if we have C11, C12, etc in metrics
    c11_metric = next(m for m in result.metrics if m.name == "C11")
    assert c11_metric.value == 100.0


def test_elasticity_validator_integration(mock_atoms, validation_config, potential_config, tmp_path):
    """Test full execution without mocking internal method (for coverage)."""
    validator = ElasticityValidator(
        potential_path=tmp_path / "pot.yace",
        config=validation_config,
        potential_config=potential_config
    )

    # Still need to mock calculator to avoid LAMMPS executable dependency
    validator._get_calculator = MagicMock(return_value=MockCalculator())

    result = validator.validate(reference_structure=mock_atoms)

    # With dummy calculator (stress constant), Cij should be 0.
    # So it should FAIL Born stability.
    assert result.overall_status == "FAIL"
    c11_metric = next(m for m in result.metrics if m.name == "C11")
    assert c11_metric.value == 0.0
