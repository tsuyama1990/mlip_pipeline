from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mlip_autopipec.config.loader import load_config
from mlip_autopipec.main import create_components, main


def test_create_components(temp_dir: Path) -> None:
    # Setup Config
    config_data = {
        "project": {"name": "CompTest"},
        "training": {"dataset_path": str(temp_dir / "data.pckl"), "max_epochs": 1},
        "orchestrator": {"max_iterations": 1},
        "validation": {"run_validation": True},
        "dft": {"pseudopotentials": {"Si": "Si.upf"}},
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Touch the data file to pass validation
    (temp_dir / "data.pckl").touch()

    config = load_config(config_file)

    explorer, oracle, trainer, validator = create_components(config)

    # Check types (using structural typing or concrete mocks/classes)
    # Since interfaces are Protocols, isinstance checks against them works if they are runtime checkable
    # But Explorer is a Protocol, so we check if it implements explore
    assert hasattr(explorer, "explore")
    assert hasattr(oracle, "compute")
    assert hasattr(trainer, "train")
    assert hasattr(validator, "validate")


def test_main_integration(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    # Mock Orchestrator to avoid running full loop
    with patch("mlip_autopipec.main.Orchestrator") as MockOrch:
        instance = MockOrch.return_value

        # Call main with args
        with patch("sys.argv", ["main.py", str(config_file)]):
            main()

        # Verify Orchestrator was initialized and run
        MockOrch.assert_called_once()
        instance.run.assert_called_once()


def test_main_missing_config(temp_dir: Path) -> None:
    with patch("sys.argv", ["main.py", "non_existent.yaml"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
