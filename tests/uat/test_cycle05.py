from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


def test_uat_cycle05_01_phonon_stability_pass(tmp_path: Path) -> None:
    """
    UAT-C05-01: Verify that a stable crystal structure has no imaginary phonon modes.
    Success Criteria: Status PASS, HTML report generated.
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        (tmp_path / "pot.yace").touch()

        # Mock ValidationRunner internals to simulate PASS
        with (
            patch(
                "mlip_autopipec.physics.validation.runner.PhononValidator"
            ) as MockPhonon,
            patch(
                "mlip_autopipec.physics.validation.runner.ElasticityValidator"
            ) as MockElasticity,
            patch("mlip_autopipec.physics.validation.runner.EOSValidator") as MockEOS,
            patch(
                "mlip_autopipec.physics.validation.runner.StructureGenFactory"
            ) as MockGen,
        ):
            # Setup returns
            MockPhonon.return_value.validate.return_value = (
                [
                    MagicMock(
                        passed=True, name="Phonon Stability", value=1.0, unit="THz"
                    )
                ],
                {},
            )
            MockElasticity.return_value.validate.return_value = (
                [
                    MagicMock(
                        passed=True, name="Elastic Stability", value=1.0, unit="GPa"
                    )
                ],
                {},
            )
            MockEOS.return_value.validate.return_value = (
                [MagicMock(passed=True, name="Bulk Modulus", value=100.0, unit="GPa")],
                {},
            )
            MockGen.get_generator.return_value.generate.return_value = MagicMock()

            result = runner.invoke(app, ["validate", "--potential", "pot.yace"])

            assert result.exit_code == 0
            assert Path("validation_report.html").exists()
            assert "Validation Completed" in result.stdout

            # Verify report content
            report = Path("validation_report.html").read_text()
            assert "PASS" in report
            assert "Phonon Stability" in report


def test_uat_cycle05_02_bad_potential(tmp_path: Path) -> None:
    """
    UAT-C05-02: Force validation failure.
    Success Criteria: Status FAIL.
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        (tmp_path / "pot.yace").touch()

        with (
            patch(
                "mlip_autopipec.physics.validation.runner.PhononValidator"
            ) as MockPhonon,
            patch(
                "mlip_autopipec.physics.validation.runner.ElasticityValidator"
            ) as MockElasticity,
            patch("mlip_autopipec.physics.validation.runner.EOSValidator") as MockEOS,
            patch(
                "mlip_autopipec.physics.validation.runner.StructureGenFactory"
            ) as MockGen,
        ):
            # Setup returns: Phonon Fails
            MockPhonon.return_value.validate.return_value = (
                [
                    MagicMock(
                        passed=False, name="Phonon Stability", value=-1.0, unit="THz"
                    )
                ],
                {},
            )
            MockElasticity.return_value.validate.return_value = (
                [MagicMock(passed=True, name="Elastic", value=1.0)],
                {},
            )
            MockEOS.return_value.validate.return_value = (
                [MagicMock(passed=True, name="EOS", value=1.0)],
                {},
            )
            MockGen.get_generator.return_value.generate.return_value = MagicMock()

            result = runner.invoke(app, ["validate", "--potential", "pot.yace"])

            assert result.exit_code == 0  # Command succeeds, but report says FAIL
            report = Path("validation_report.html").read_text()
            assert "FAIL" in report
            assert "Phonon Stability" in report
