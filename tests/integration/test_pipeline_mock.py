"""Integration tests for the full active learning pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
    TrainerConfig,
)
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def integration_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PYACEMAKERConfig:
    """Configuration for integration tests."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)

    # Create fake pseudopotential
    pp_path = tmp_path / "H.pbe.UPF"
    pp_path.touch()

    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="IntegrationTest", root_dir=tmp_path),
        orchestrator={
            "max_cycles": 2,
            "n_local_candidates": 2,
            "n_active_set_select": 2,
            # Validation split ensures some data goes to validation
            "validation_split": 0.2,
            "min_validation_size": 1
        },
        oracle=OracleConfig(
            dft=DFTConfig(pseudopotentials={"H": str(pp_path)}),
            mock=True  # Use MockOracle
        ),
        trainer=TrainerConfig(mock=True), # Use mock trainer logic (file generation)
        structure_generator=StructureGeneratorConfig(strategy="random"),
    )


def test_full_cycle_execution(integration_config: PYACEMAKERConfig) -> None:
    """Test full execution of the orchestrator with mocked modules."""

    # We rely on default mocked modules instantiated by Orchestrator
    # But we need to ensure the Trainer works with files.
    # The default PacemakerTrainer with mock=True skips actual CLI calls but does file IO.

    orchestrator = Orchestrator(config=integration_config)

    # Mock Validator to always pass
    orchestrator.validator = MagicMock()
    orchestrator.validator.validate.return_value = MagicMock(status="success", metrics={})

    # Mock Dynamics Engine to always find high uncertainty structures
    from unittest.mock import patch
    with patch("pyacemaker.modules.dynamics_engine.LAMMPSEngine._simulate_halt_condition", return_value=True):
        # Run
        result = orchestrator.run()

    assert result.status == "success"
    # Access Pydantic model extra fields via model_dump or getattr
    metrics_dict = result.metrics.model_dump()
    assert metrics_dict["cycles"] == 2

    # Verify data files created
    data_dir = integration_config.project.root_dir / "data"
    dataset_path = data_dir / "dataset.pckl.gzip"
    assert dataset_path.exists()

    # Verify size
    # We can check if it has content
    assert dataset_path.stat().st_size > 0
