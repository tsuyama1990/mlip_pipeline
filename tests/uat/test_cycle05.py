from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mlip_autopipec.app import app
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult

runner = CliRunner()

def test_uat_cycle05_validate_command_pass(tmp_path):
    # Create dummy files
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    potential_file = tmp_path / "potential.yace"
    potential_file.touch()

    with patch("mlip_autopipec.cli.commands.ValidationRunner") as MockRunner, \
         patch("mlip_autopipec.cli.commands.Config.from_yaml") as MockConfigLoad, \
         patch("mlip_autopipec.cli.commands.logging_infra.setup_logging"):

         mock_runner_inst = MockRunner.return_value
         mock_runner_inst.validate.return_value = ValidationResult(
             potential_id="test",
             metrics=[],
             plots={},
             overall_status="PASS"
         )

         # Mock Config object
         mock_config = MagicMock()
         mock_config.validation = ValidationConfig()
         mock_config.potential = PotentialConfig(elements=["Si"], cutoff=5.0)
         mock_config.lammps.command = "lammps"
         MockConfigLoad.return_value = mock_config

         result = runner.invoke(app, ["validate", "--config", str(config_file), "--potential", str(potential_file)])

         assert result.exit_code == 0
         assert "Validation Completed: PASS" in result.stdout

def test_uat_cycle05_validate_command_warn(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    potential_file = tmp_path / "potential.yace"
    potential_file.touch()

    with patch("mlip_autopipec.cli.commands.ValidationRunner") as MockRunner, \
         patch("mlip_autopipec.cli.commands.Config.from_yaml") as MockConfigLoad, \
         patch("mlip_autopipec.cli.commands.logging_infra.setup_logging"):

         mock_runner_inst = MockRunner.return_value
         mock_runner_inst.validate.return_value = ValidationResult(
             potential_id="test",
             metrics=[],
             plots={},
             overall_status="WARN"
         )

         mock_config = MagicMock()
         mock_config.validation = ValidationConfig()
         mock_config.potential = PotentialConfig(elements=["Si"], cutoff=5.0)
         mock_config.lammps.command = "lammps"
         MockConfigLoad.return_value = mock_config

         result = runner.invoke(app, ["validate", "--config", str(config_file), "--potential", str(potential_file)])

         assert result.exit_code == 0
         assert "Validation Completed: WARN" in result.stdout

def test_uat_cycle05_validate_command_fail(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    potential_file = tmp_path / "potential.yace"
    potential_file.touch()

    with patch("mlip_autopipec.cli.commands.ValidationRunner") as MockRunner, \
         patch("mlip_autopipec.cli.commands.Config.from_yaml") as MockConfigLoad, \
         patch("mlip_autopipec.cli.commands.logging_infra.setup_logging"):

         mock_runner_inst = MockRunner.return_value
         mock_runner_inst.validate.return_value = ValidationResult(
             potential_id="test",
             metrics=[],
             plots={},
             overall_status="FAIL"
         )

         mock_config = MagicMock()
         mock_config.validation = ValidationConfig()
         mock_config.potential = PotentialConfig(elements=["Si"], cutoff=5.0)
         mock_config.lammps.command = "lammps"
         MockConfigLoad.return_value = mock_config

         result = runner.invoke(app, ["validate", "--config", str(config_file), "--potential", str(potential_file)])

         assert result.exit_code == 1
         assert "Validation Failed" in result.stdout
