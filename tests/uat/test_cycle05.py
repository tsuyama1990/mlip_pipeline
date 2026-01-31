import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


def setup_config(path: Path):
    config_content = """
project_name: "UAT_Project_C05"
potential:
  elements: ["Al"]
  cutoff: 5.0
structure_gen:
  strategy: "bulk"
  element: "Al"
  crystal_structure: "fcc"
  lattice_constant: 4.05
md:
  temperature: 300.0
  n_steps: 10
  timestep: 0.001
  ensemble: "NVT"
lammps:
  command: "lmp"
validation:
  phonon_tolerance: -0.1
  eos_vol_range: 0.1
  eos_n_points: 5
  elastic_strain: 0.01
"""
    (path / "config.yaml").write_text(config_content)
    # Mock potential file
    (path / "potential.yace").write_text("Mock Potential Content")


def test_uat_c05_01_validation_success(tmp_path):
    """
    Scenario 5.1: Validate a stable potential
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td))

        # We mock the high-level components since we are testing the CLI wiring here
        # and we don't want to run actual heavy physics calculations in UAT if possible,
        # or at least we want to control the outcome.
        with (
            patch(
                "mlip_autopipec.physics.validation.runner.ValidationRunner.validate"
            ) as mock_validate,
        ):
            from mlip_autopipec.domain_models.validation import (
                ValidationResult,
                ValidationMetric,
            )

            # Mock successful validation result
            mock_validate.return_value = ValidationResult(
                potential_id="potential.yace",
                metrics=[
                    ValidationMetric(name="Phonon Stability", value=0.0, passed=True),
                    ValidationMetric(
                        name="Bulk Modulus", value=70.0, reference=76.0, passed=True
                    ),
                ],
                overall_status="PASS",
            )

            result = runner.invoke(
                app,
                [
                    "validate",
                    "--config",
                    "config.yaml",
                    "--potential",
                    "potential.yace",
                ],
            )

            assert result.exit_code == 0
            assert "Validation Completed: Status PASS" in result.stdout
            assert "Report generated" in result.stdout
            mock_validate.assert_called_once()


def test_uat_c05_02_validation_failure(tmp_path):
    """
    Scenario 5.2: Catching a Bad Potential
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td))

        with (
            patch(
                "mlip_autopipec.physics.validation.runner.ValidationRunner.validate"
            ) as mock_validate,
        ):
            from mlip_autopipec.domain_models.validation import (
                ValidationResult,
                ValidationMetric,
            )

            # Mock failed validation result
            mock_validate.return_value = ValidationResult(
                potential_id="potential.yace",
                metrics=[
                    ValidationMetric(name="Phonon Stability", value=-0.5, passed=False)
                ],
                overall_status="FAIL",
            )

            result = runner.invoke(
                app,
                [
                    "validate",
                    "--config",
                    "config.yaml",
                    "--potential",
                    "potential.yace",
                ],
            )

            assert result.exit_code == 0
            assert "Validation Completed: Status FAIL" in result.stdout
            mock_validate.assert_called_once()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
