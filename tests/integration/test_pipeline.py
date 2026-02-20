"""Integration tests for the PYACEMAKER pipeline."""

from pathlib import Path

import yaml

from pyacemaker.core.config import CONSTANTS
from pyacemaker.core.config_loader import load_config
from pyacemaker.orchestrator import Orchestrator


def test_pipeline_integration(tmp_path: Path) -> None:
    """Test running the orchestrator with a loaded configuration and mock modules."""
    # Skip file checks for this test
    CONSTANTS.skip_file_checks = True

    # Create a valid configuration file
    config_data = {
        "version": "0.1.0",
        "project": {"name": "IntegrationTest", "root_dir": str(tmp_path)},
        "oracle": {
            "dft": {
                "code": "quantum_espresso",
                "pseudopotentials": {"Fe": "Fe.pbe.UPF"},
            },
            "mock": True,
        },
        "trainer": {
            "mock": True,
        },
        "orchestrator": {
            "max_cycles": 2,
            "uncertainty_threshold": 0.1,
            "n_local_candidates": 5,
            "n_active_set_select": 2,
        },
    }

    config_file = tmp_path / "integration_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Load configuration
    config = load_config(config_file)

    # Initialize Orchestrator (it uses default mock modules if none provided)
    orchestrator = Orchestrator(config)

    # We need to mock the validator because real validator runs checks on mock data which might fail
    # or just use MockValidator if we want pure pipeline test.
    # But Orchestrator uses Validator by default now.
    # Validator will try to run check_phonons on empty/mock atoms and likely fail or return false.
    # The default Validator implementation tries to validate.
    # If the validation fails, the cycle fails.
    # So we should inject a MockValidator that always passes for this integration test.

    from unittest.mock import MagicMock

    from pyacemaker.core.base import Metrics, ModuleResult

    mock_validator = MagicMock()
    mock_validator.validate.return_value = ModuleResult(
        status="success", metrics=Metrics(), artifacts={}
    )
    orchestrator.validator = mock_validator

    # Run pipeline
    result = orchestrator.run()

    # Verify success
    assert result.status == "success"
    # Metrics model uses extra='allow', so attributes are dynamic.
    # We use model_dump to access them safely.
    metrics = result.metrics.model_dump()
    assert metrics["cycles"] > 0
