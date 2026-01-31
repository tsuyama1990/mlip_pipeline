from unittest.mock import patch
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.physics.validation.runner import ValidationRunner

@pytest.fixture
def mock_atoms():
    return Atoms("Al", positions=[[0, 0, 0]], cell=[[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]], pbc=True)

@pytest.fixture
def validation_config():
    return ValidationConfig()

@pytest.fixture
def potential_config():
    return PotentialConfig(elements=["Al"], cutoff=5.0)

def test_validation_runner(mock_atoms, validation_config, potential_config, tmp_path):
    runner = ValidationRunner(
        potential_path=tmp_path / "pot.yace",
        config=validation_config,
        potential_config=potential_config
    )

    # Mock validators
    with patch("mlip_autopipec.physics.validation.runner.EOSValidator") as MockEOS, \
         patch("mlip_autopipec.physics.validation.runner.ElasticityValidator") as MockElastic, \
         patch("mlip_autopipec.physics.validation.runner.PhononValidator") as MockPhonon:

        MockEOS.return_value.validate.return_value = ValidationResult(
            potential_id="test", metrics=[], plots={}, overall_status="PASS"
        )
        MockElastic.return_value.validate.return_value = ValidationResult(
            potential_id="test", metrics=[], plots={}, overall_status="PASS"
        )
        MockPhonon.return_value.validate.return_value = ValidationResult(
            potential_id="test", metrics=[], plots={}, overall_status="PASS"
        )

        results = runner.validate(reference_structure=mock_atoms)

        assert len(results) == 3
        # Ensure all sub-validators were called
        MockEOS.return_value.validate.assert_called_once()
        MockElastic.return_value.validate.assert_called_once()
        MockPhonon.return_value.validate.assert_called_once()
