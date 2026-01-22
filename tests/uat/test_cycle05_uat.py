import pytest
import os
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch

from mlip_autopipec.app import app
from mlip_autopipec.config.schemas.training import TrainingConfig

runner = CliRunner()

@pytest.fixture
def mock_db_manager():
    with patch("mlip_autopipec.core.database.DatabaseManager") as MockDB:
        instance = MockDB.return_value
        instance.__enter__.return_value = instance
        yield instance

@pytest.fixture
def mock_pacemaker():
    with patch("mlip_autopipec.training.pacemaker.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run

def test_scenario_5_1_data_preparation(mock_db_manager, tmp_path):
    """
    Scenario 5.1: Data Preparation for Training
    """
    # Setup mock data
    mock_db_manager.get_atoms.return_value = [MagicMock() for _ in range(100)]

    # Ensure pseudo dir exists
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()

    config_content = f"""
target_system:
  name: "Al"
  elements: ["Al"]
  composition: {{"Al": 1.0}}

dft:
  ecutwfc: 40.0
  kspacing: 0.04
  pseudopotential_dir: "{str(pseudo_dir)}"

training_config:
  cutoff: 5.0
  b_basis_size: 100
  kappa: 0.5
  kappa_f: 100.0
  max_iter: 100
"""

    with patch("mlip_autopipec.training.dataset.DatasetBuilder") as MockBuilder:
        builder_instance = MockBuilder.return_value

        with runner.isolated_filesystem():
            with open("mlip.yaml", "w") as f:
                f.write(config_content)

            # Re-create pseudo dir inside isolated filesystem if it's different
            # Typer runner isolated_filesystem uses a temp dir.
            # But tmp_path is another temp dir.
            # config_content refers to tmp_path / "pseudo".
            # This works because path is absolute.

            with patch("mlip_autopipec.app.DatabaseManager") as MockDBApp:
                instance = MockDBApp.return_value
                instance.__enter__.return_value = instance

                result = runner.invoke(app, ["train", "--prepare-only", "--config", "mlip.yaml"])

                if result.exit_code != 0:
                    print(result.stdout)

                assert result.exit_code == 0
                builder_instance.export.assert_called_once()


def test_scenario_5_2_training_execution(mock_db_manager, mock_pacemaker, tmp_path):
    """
    Scenario 5.2: Training Execution
    """
    # Ensure pseudo dir exists
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir(exist_ok=True)

    config_content = f"""
target_system:
  name: "Al"
  elements: ["Al"]
  composition: {{"Al": 1.0}}

dft:
  ecutwfc: 40.0
  kspacing: 0.04
  pseudopotential_dir: "{str(pseudo_dir)}"

training_config:
  cutoff: 5.0
  b_basis_size: 100
  kappa: 0.5
  kappa_f: 100.0
  max_iter: 100
"""

    with patch("mlip_autopipec.training.dataset.DatasetBuilder"), \
         patch("mlip_autopipec.training.pacemaker.PacemakerWrapper") as MockWrapper, \
         patch("mlip_autopipec.app.DatabaseManager") as MockDBApp:

        instance = MockDBApp.return_value
        instance.__enter__.return_value = instance

        wrapper_instance = MockWrapper.return_value
        wrapper_instance.train.return_value.success = True
        wrapper_instance.train.return_value.metrics = MagicMock()
        wrapper_instance.train.return_value.potential_path = "pot.yace"

        with runner.isolated_filesystem():
            with open("mlip.yaml", "w") as f:
                f.write(config_content)

            result = runner.invoke(app, ["train", "--config", "mlip.yaml"])

            if result.exit_code != 0:
                 print(result.stdout)

            assert result.exit_code == 0
            wrapper_instance.train.assert_called_once()
