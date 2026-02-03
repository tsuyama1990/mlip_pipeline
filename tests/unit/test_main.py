from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.main import main


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
    with patch("sys.argv", ["main", "ghost.yaml"]), \
         patch("sys.stderr"):

        with pytest.raises(SystemExit) as exc:
             main()
    assert exc.value.code == 1


def test_main_success(valid_config_yaml: Path) -> None:
    with patch("sys.argv", ["main", str(valid_config_yaml)]), \
         patch("mlip_autopipec.main.load_config") as mock_load, \
         patch("mlip_autopipec.main.create_components") as mock_create, \
         patch("mlip_autopipec.main.Orchestrator") as MockOrch:

        # Setup mocks
        mock_orch_instance = MockOrch.return_value

        # FIX: Set return value for create_components
        mock_create.return_value = (
            MagicMock(), # explorer
            MagicMock(), # selector
            MagicMock(), # oracle
            MagicMock(), # trainer
            MagicMock(), # validator
        )

        main()

        mock_load.assert_called_once()
        mock_create.assert_called_once()
        MockOrch.assert_called_once()
        mock_orch_instance.run.assert_called_once()


def test_main_exception(valid_config_yaml: Path) -> None:
    # PT012: Ensure pytest.raises only wraps the call to main()
    with patch("sys.argv", ["main", str(valid_config_yaml)]), \
         patch("mlip_autopipec.main.Orchestrator") as MockOrch, \
         patch("sys.stderr"), \
         patch("mlip_autopipec.main.create_components") as mock_create:

        # Also need to mock create_components here otherwise it might fail unpacking if called before exception?
        # main() calls create_components before Orchestrator init.
        mock_create.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        MockOrch.side_effect = Exception("Boom")

        with pytest.raises(SystemExit) as exc:
             main()

        assert exc.value.code == 1
