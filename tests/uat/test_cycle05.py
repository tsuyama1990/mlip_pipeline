import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric

runner = CliRunner()


@pytest.fixture
def mock_calc():
    """Mock the LAMMPS calculator."""
    with patch("mlip_autopipec.physics.validation.utils.get_lammps_calculator") as mock:
        calc = MagicMock()
        # Mocking get_potential_energy, get_stress, get_forces
        calc.get_potential_energy.return_value = -100.0
        calc.get_stress.return_value = np.zeros(6)
        calc.get_forces.return_value = np.zeros((2, 3))

        # When get_lammps_calculator is called, return this calc
        mock.return_value = calc
        yield calc


def test_uat_cycle05_validate_flow(tmp_path, mock_calc):
    # Change CWD to tmp_path to isolate files
    current_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        # 1. Init Project
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0

        # 2. Mock a potential file
        potential_path = tmp_path / "potential.yace"
        potential_path.touch()

        # 3. Mock ValidationRunner components (Phonon/EOS/Elasticity) to avoid heavy computation
        # We'll patch ValidationRunner.validate directly to return a passed result
        # This tests the CLI integration mostly.

        with patch("mlip_autopipec.physics.validation.runner.ValidationRunner.validate") as mock_validate:
            mock_validate.return_value = ValidationResult(
                potential_id="potential.yace",
                metrics=[
                    ValidationMetric(name="EOS", value=0.01, passed=True),
                    ValidationMetric(name="Elasticity", value=0.01, passed=True),
                    ValidationMetric(name="Phonon", value=0.0, passed=True)
                ],
                plots={},
                overall_status="PASS"
            )

            # 4. Run Validate Command
            result = runner.invoke(app, ["validate", "--config", "config.yaml", "--potential", "potential.yace"])

            assert result.exit_code == 0
            assert "Validation Finished: PASS" in result.stdout
            assert "EOS: 0.0100" in result.stdout
            assert "Report generated" in result.stdout

    finally:
        os.chdir(current_dir)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
