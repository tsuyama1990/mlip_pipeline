"""Integration tests for the PYACEMAKER pipeline."""

from pathlib import Path

import yaml

from pyacemaker.core.config import load_config
from pyacemaker.orchestrator import Orchestrator


def test_pipeline_integration(tmp_path: Path) -> None:
    """Test running the orchestrator with a loaded configuration and mock modules."""
    # Create a valid configuration file
    config_data = {
        "version": "0.1.0",
        "project": {"name": "IntegrationTest", "root_dir": str(tmp_path)},
        "oracle": {"dft": {"code": "quantum_espresso"}, "mock": True},
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

    # Run pipeline
    result = orchestrator.run()

    # Verify success
    assert result.status == "success"
    # Metrics model uses extra='allow', so attributes are dynamic.
    # We use model_dump to access them safely.
    metrics = result.metrics.model_dump()
    assert metrics["cycles"] > 0
    assert metrics["dataset_size"] > 0
