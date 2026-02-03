from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config.loader import load_config
from mlip_autopipec.main import create_components, main
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def valid_config_yaml(tmp_path: Path) -> Path:
    # Ensure dataset exists
    data_path = tmp_path / "data.pckl"
    data_path.touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
project:
  name: "TestProject"
training:
  dataset_path: "{data_path}"
  max_epochs: 10
orchestrator:
  max_iterations: 1
exploration:
  strategy: "adaptive"
selection:
  method: "random"
  max_structures: 10
oracle:
  method: "mock"
validation:
  run_validation: false
""")
    return config_path


def test_main_no_config() -> None:
    with (
        patch("sys.argv", ["main", "ghost.yaml"]),
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 1


def test_main_success(valid_config_yaml: Path) -> None:
    with (
        patch("sys.argv", ["main", str(valid_config_yaml)]),
        patch("mlip_autopipec.main.load_config") as mock_load,
        patch("mlip_autopipec.main.create_components") as mock_create,
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
    ):
        # Setup mocks
        mock_orch_instance = MockOrch.return_value
        # mock_create must return 5 values to unpack
        mock_create.return_value = (None, None, None, None, None)

        main()

        mock_load.assert_called_once()
        mock_create.assert_called_once()
        MockOrch.assert_called_once()
        mock_orch_instance.run.assert_called_once()


def test_main_exception(valid_config_yaml: Path) -> None:
    with (
        patch("sys.argv", ["main", str(valid_config_yaml)]),
        patch("mlip_autopipec.main.Orchestrator") as MockOrch,
    ):
        MockOrch.side_effect = Exception("Boom")

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1


def test_create_components_wiring(valid_config_yaml: Path) -> None:
    # 1. Test with validation DISABLED
    config = load_config(valid_config_yaml)

    explorer, selector, oracle, trainer, validator = create_components(config)
    assert validator is None

    # 2. Test with validation ENABLED
    config.validation.run_validation = True
    explorer, selector, oracle, trainer, validator = create_components(config)

    assert isinstance(validator, ValidationRunner)
