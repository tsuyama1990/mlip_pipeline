from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mlip_autopipec.cli import main


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
    with patch("mlip_autopipec.cli.Orchestrator") as MockOrch:
        instance = MockOrch.return_value

        # Call main with args
        with patch("sys.argv", ["cli.py", str(config_file)]):
            main()

        # Verify Orchestrator was initialized and run
        MockOrch.assert_called_once()
        instance.run.assert_called_once()


def test_cli_missing_config(temp_dir: Path) -> None:
    with patch("sys.argv", ["cli.py", "non_existent.yaml"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
