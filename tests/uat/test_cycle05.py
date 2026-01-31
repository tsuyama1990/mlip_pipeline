from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric

runner = CliRunner()


def test_uat_c05_validate_command():
    # Scenario: Validate a potential

    with (
        patch("mlip_autopipec.cli.commands.ValidationRunner") as mock_runner_cls,
        patch("mlip_autopipec.cli.commands.Config.from_yaml") as mock_config_loader,
        patch("mlip_autopipec.cli.commands.logging_infra.setup_logging"),
        patch(
            "mlip_autopipec.physics.structure_gen.generator.StructureGenFactory"
        ) as mock_gen_factory,
    ):
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        mock_gen = MagicMock()
        mock_gen.generate.return_value = MagicMock(get_chemical_formula=lambda: "Si")
        mock_gen_factory.get_generator.return_value = mock_gen

        # Mock result
        res = ValidationResult(
            potential_id="test",
            metrics=[
                ValidationMetric(name="Phonon Stability", value=1.0, passed=True),
                ValidationMetric(
                    name="Bulk Modulus",
                    value=100.0,
                    reference=98.0,
                    error=2.0,
                    passed=True,
                ),
            ],
            plots={"phonon": Path("phonon.png")},
            overall_status="PASS",
        )
        mock_runner.validate.return_value = res

        # Mock config
        mock_config = MagicMock()
        mock_config.validation.report_path = Path("validation_report.html")
        # Ensure potential config exists
        mock_config.potential = MagicMock()
        mock_config.structure_gen.strategy = "bulk"
        mock_config_loader.return_value = mock_config

        # Create dummy config file and potential
        with runner.isolated_filesystem():
            Path("config.yaml").touch()
            Path("potential.yace").touch()

            result = runner.invoke(
                app, ["validate", "--config", "config.yaml", "--potential", "potential.yace"]
            )

            assert result.exit_code == 0
            assert "Validation Completed" in result.stdout
            assert "Status: PASS" in result.stdout
