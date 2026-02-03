from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlip_autopipec.cli import create_components, main
from mlip_autopipec.config.loader import load_config
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.physics.oracle.manager import DFTManager


def test_cli_integration(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup Config
    config_data = {
        "project": {"name": "CLI_Test"},
        "training": {"dataset_path": str(temp_dir / "data.pckl"), "max_epochs": 1},
        "orchestrator": {"max_iterations": 1},
        "validation": {"run_validation": False},
        "dft": {"pseudopotentials": {"Si": "Si.upf"}},
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    (temp_dir / "data.pckl").touch()

    # Mock Orchestrator to avoid running full loop (we test loop in test_orchestrator)
    # We also need to mock create_components if it exists, or rely on it working if we implement it.
    # Since we haven't implemented it yet, this test might fail if main() changes signature or logic drastically.
    # But main() calls Orchestrator.

    with (
        patch("mlip_autopipec.cli.Orchestrator") as MockOrch,
        patch("mlip_autopipec.cli.create_components") as MockFactory,
    ):
        mock_explorer = MagicMock()
        mock_oracle = MagicMock()
        mock_trainer = MagicMock()
        mock_validator = MagicMock()
        MockFactory.return_value = (
            mock_explorer,
            mock_oracle,
            mock_trainer,
            mock_validator,
        )

        instance = MockOrch.return_value

        # Call main with args
        with patch("sys.argv", ["cli.py", str(config_file)]):
            main()

        # Verify Orchestrator was initialized and run
        MockOrch.assert_called_once()
        instance.run.assert_called_once()
        MockFactory.assert_called_once()


def test_cli_missing_config(temp_dir: Path) -> None:
    with patch("sys.argv", ["cli.py", "non_existent.yaml"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1


def test_create_components(temp_dir: Path) -> None:
    # Ensure data file exists for Pydantic validation
    data_file = temp_dir / "data.pckl"
    data_file.touch()

    # 1. Test with Mock Oracle and Validator
    config_data: dict[str, Any] = {
        "project": {"name": "Test"},
        "training": {"dataset_path": str(data_file), "max_epochs": 1},
        "orchestrator": {"max_iterations": 1},
        "validation": {"run_validation": True},
        "oracle": {"method": "mock"},
    }
    config_file = temp_dir / "config_mock.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    explorer, oracle, trainer, validator = create_components(config)

    assert isinstance(explorer, MockExplorer)
    assert isinstance(oracle, MockOracle)
    assert isinstance(validator, MockValidator)

    # 2. Test with DFT Oracle and No Validator
    config_data["oracle"]["method"] = "dft"
    config_data["dft"] = {"pseudopotentials": {"Si": "Si.upf"}}
    config_data["validation"]["run_validation"] = False

    config_file_dft = temp_dir / "config_dft.yaml"
    with config_file_dft.open("w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file_dft)
    explorer, oracle, trainer, validator = create_components(config)

    assert isinstance(explorer, MockExplorer)
    assert isinstance(oracle, DFTManager)
    assert validator is None
