from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mlip_autopipec.main import main


@pytest.fixture
def valid_config_yaml(temp_dir: Path) -> Path:
    data_file = temp_dir / "data.pckl"
    data_file.touch()

    config_data = {
        "project": {"name": "TestProject"},
        "training": {"dataset_path": str(data_file)},
        "orchestrator": {"max_iterations": 1},
        "oracle": {"method": "mock"},
        "validation": {"run_validation": False},
        "exploration": {"strategy": "adaptive"},
    }
    config_file = temp_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)
    return config_file


def test_main_no_config() -> None:
    with (
        patch("sys.argv", ["main", "ghost.yaml"]),
        patch("sys.stderr"),
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 1


def test_main_exception(valid_config_yaml: Path) -> None:
    # 1. Patches setup
    with (
        patch("sys.argv", ["main", str(valid_config_yaml)]),
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
        patch("mlip_autopipec.main.create_components"),
        patch("sys.stderr"),
    ):
        MockOrch.side_effect = Exception("Boom")

        # 2. Assert exception raises SystemExit
        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1


def test_main_success(valid_config_yaml: Path) -> None:
    with (
        patch("sys.argv", ["main", str(valid_config_yaml)]),
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
        patch("mlip_autopipec.main.create_components") as MockCreate,
    ):
        mock_instance = MockOrch.return_value
        # Mock create_components return
        MockCreate.return_value = ("exp", "ora", "tra", "val")

        main()

        MockCreate.assert_called_once()
        mock_instance.run.assert_called_once()
