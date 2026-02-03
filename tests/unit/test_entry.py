from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.entry import main


@pytest.fixture
def valid_config_yaml(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
project:
  name: "TestProject"
training:
  dataset_path: "data.pckl"
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
    # Ensure no unused variables and correct exception handling
    with (
        patch("sys.argv", ["entry", "ghost.yaml"]),
        patch("sys.stderr"),
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 1


def test_main_success(valid_config_yaml: Path) -> None:
    with (
        patch("sys.argv", ["entry", str(valid_config_yaml)]),
        patch("mlip_autopipec.entry.load_config") as mock_load,
        patch("mlip_autopipec.entry.create_components") as mock_create,
        patch("mlip_autopipec.entry.Orchestrator") as MockOrch,
    ):
        # Setup mocks
        mock_orch_instance = MockOrch.return_value
        # mock_create must return 5 values to unpack (explorer, selector, oracle, trainer, validator)
        mock_create.return_value = (None, None, None, None, None)

        main()

        mock_load.assert_called_once()
        mock_create.assert_called_once()
        MockOrch.assert_called_once()
        mock_orch_instance.run.assert_called_once()


def test_main_exception(valid_config_yaml: Path) -> None:
    with (
        patch("sys.argv", ["entry", str(valid_config_yaml)]),
        patch("mlip_autopipec.entry.Orchestrator") as MockOrch,
        patch("sys.stderr"),
    ):
        MockOrch.side_effect = Exception("Boom")

        # Nested pytest.raises to satisfy PT012
        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
